from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
from typing import Set, Dict
import numpy as np
import random
import bisect

class EmpiricalCdf:
    def __init__(self, data: np.ndarray):
        self.x = np.sort(data)
        self.n = len(self.x)
    def __call__(self, q: float) -> float:
        idx = bisect.bisect_right(self.x, q)
        return idx / self.n

class EmpiricalPdf:
    def __init__(self, data: np.ndarray, bins: int = 200):
        hist, edges = np.histogram(data, bins=bins, density=True)
        self.hist      = hist
        self.bin_edges = edges
    def __call__(self, q: float) -> float:
        i = np.searchsorted(self.bin_edges, q, 'right') - 1
        return float(self.hist[i]) if 0 <= i < len(self.hist) else 0.0
    
class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):
    

    def __init__(self):
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = "chillguyTest"  # TODO: enter a name.
        self._campaign_bid_cache = {}
        SEG_FREQ_3 = {
        MarketSegment({'Male',   'Young',    'LowIncome'}): 1836,
        MarketSegment({'Male',   'Young',    'HighIncome'}):  517,
        MarketSegment({'Male',   'Old',      'LowIncome'}): 1795,
        MarketSegment({'Male',   'Old',      'HighIncome'}):  808,
        MarketSegment({'Female', 'Young',    'LowIncome'}): 1980,
        MarketSegment({'Female', 'Young',    'HighIncome'}):  256,
        MarketSegment({'Female', 'Old',      'LowIncome'}): 2401,
        MarketSegment({'Female', 'Old',      'HighIncome'}):  407,
        }
        # (Gender, Age)
        SEG_FREQ_2_GA = {
            MarketSegment({'Male',   'Young'}): 2353,
            MarketSegment({'Male',   'Old'}):   2603,
            MarketSegment({'Female', 'Young'}): 2236,
            MarketSegment({'Female', 'Old'}):   2808,
        }

        # (Gender, Income)
        SEG_FREQ_2_GI = {
            MarketSegment({'Male',   'LowIncome'}):  3631,
            MarketSegment({'Male',   'HighIncome'}): 1325,
            MarketSegment({'Female', 'LowIncome'}):  4381,
            MarketSegment({'Female', 'HighIncome'}):  663,
        }

        # (Age, Income)
        SEG_FREQ_2_AI = {
            MarketSegment({'Young', 'LowIncome'}): 3816,
            MarketSegment({'Young', 'HighIncome'}): 773,
            MarketSegment({'Old',   'LowIncome'}): 4196,
            MarketSegment({'Old',   'HighIncome'}):1215,
        }
        SEG_FREQ_1 = {
            # by Gender
            MarketSegment({'Male'}):   4956,
            MarketSegment({'Female'}): 5044,
            # by Age
            MarketSegment({'Young'}):  4589,
            MarketSegment({'Old'}):    5411,
            # by Income
            MarketSegment({'LowIncome'}):  8012,
            MarketSegment({'HighIncome'}): 1988,
        }
        self.SEG_FREQ = {}
        self.SEG_FREQ.update(SEG_FREQ_1)
        self.SEG_FREQ.update(SEG_FREQ_2_GA)
        self.SEG_FREQ.update(SEG_FREQ_2_GI)
        self.SEG_FREQ.update(SEG_FREQ_2_AI)
        self.SEG_FREQ.update(SEG_FREQ_3)
        self.prior_cdf: Dict[MarketSegment, EmpiricalCdf] = {}
        self.prior_pdf: Dict[MarketSegment, EmpiricalPdf] = {}
        for seg, avg in self.SEG_FREQ.items():
            # sample many opponent effective bids
            pool = []
            for _ in range(2000):
                D = random.choice([1,2,3])
                delta = random.choice([0.3,0.5,0.7])
                R_i = avg * D * delta
                # opponents bid Uniform[0.1Ri, Ri], quality≈1 early‐game
                b_i = random.uniform(0.1*R_i, R_i) 
                pool.append(b_i)
            arr = np.array(pool)
            self.prior_cdf[seg] = EmpiricalCdf(arr)
            self.prior_pdf[seg] = EmpiricalPdf(arr)

    def on_new_game(self) -> None:
        # TODO: fill this in (if necessary)
        self._campaign_bid_cache.clear()
        #pass

    def optimize_b_reverse_2nd_price(self,R: float,
                                 cdf: EmpiricalCdf,
                                 pdf: EmpiricalPdf,
                                 m: int = 10) -> float:
        """grid seaech using cdf pdf"""
        L, U = 0.1 * R, R
        grid = np.linspace(L, U, 100)
        best_b, best_val = L, -1e9

        for b in grid:
            # Probability we win = [F(b)]^(m-1)
            F_b = cdf(b)
            p_win = F_b ** (m - 1)

            # build xs for the integral
            xs = np.linspace(L, b, 100)

            # vectorize CDF/pdf calls
            Fxs = np.array([cdf(xi) for xi in xs])
            fxs = np.array([pdf(xi) for xi in xs])

            # weights for the second-lowest order-statistic
            # f_{(2)}(x) = (m-1) * [F(x)]^(m-2) * f(x)
            weights = (m - 1) * (Fxs ** (m - 2)) * fxs

            # expected second-lowest conditional on our bid = b
            denom = max(F_b ** (m - 1), 1e-12)
            sec_low = np.trapezoid(xs * weights, xs) / denom

            # revenue ≈ R (since ρ(R)=1 at full reach)
            val = p_win * (R - sec_low)
            if val > best_val:
                best_val, best_b = val, b

        return float(np.clip(best_b, L, U))


    # def _best_campaign_bid(self, R: float) -> float:
    #     """
    #     Numerically maximize the expected‐profit function for a campaign of reach R
    #     under the uniform‐bid assumption.
    #     """
    #     if R in self._campaign_bid_cache:
    #         return self._campaign_bid_cache[R]

    #     L = 0.1 * R
    #     U = R
    #     grid = np.linspace(L, U, 200, endpoint=True)
    #     # Precompute constants
    #     # revenue at full reach:
    #     rev = R * self.effective_reach(R, R)  # exactly = R
    #     best_b, best_val = L, -1e9

    #     for b in grid:
    #         # Pr{win} = ((b - L)/(U - L))^9
    #         p_win = ((b - L) / (U - L)) ** (self.num_agents() - 1)
    #         # E[second_lowest | we bid = b] = b + (U - b)/m_total
    #         cost_if_win = b + (U - b) / self.num_agents()
    #         val = p_win * (rev - cost_if_win)
    #         if val > best_val:
    #             best_val, best_b = val, b

    #     # clip into valid range and cache
    #     clipped = float(self.clip_campaign_bid(
    #         Campaign(start_day=0, end_day=0, target=frozenset(), reach=R),
    #         best_b
    #     ))
    #     self._campaign_bid_cache[R] = clipped
    #     return clipped

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        today = self.get_current_day()

        for c in self.get_active_campaigns():
            spent   = self.get_cumulative_cost(c)
            done    = self.get_cumulative_reach(c)
            remB    = max(0.0, c.budget  - spent)
            remR    = max(1, c.reach   - done)

            # PHASE 1: days 1–7 → finish short campaigns ASAP
            if today <= 7:
                if c.end_day <= 7:
                    # pour all remaining budget into this campaign today
                    # approximate derivative of effective reach at x impressions:
                    rho_x   = self.effective_reach(done,   c.reach)
                    rho_x1  = self.effective_reach(done+1, c.reach)
                    marginal_value = (rho_x1 - rho_x) * c.budget
                    bid_per_item = min(marginal_value, remB/remR)
                    daily_limit  = remB
                    # build the "finish it today" bundle in phase 1
                    bid_entries = set()
                    total_freq  = sum(self.SEG_FREQ[s] for s in self.SEG_FREQ
                                    if c.target_segment.issubset(s))
                    for seg, freq in self.SEG_FREQ.items():
                        if c.target_segment.issubset(seg):
                            w   = freq / total_freq
                            Ls  = daily_limit * w
                            bid_per_item = bid_per_item* w
                            # same high per‐item bid everywhere to exhaust budget
                            bid_entries.add(Bid(self, seg,
                                                bid_per_item=bid_per_item,
                                                bid_limit=Ls))

                    bundles.add(BidBundle(c.uid, limit=daily_limit,
                                        bid_entries=bid_entries))
                else:
                    continue

            # PHASE 2: days 8–10 → profit pacing (using SEG_FREQ weights)
            else:
                if c.end_day > 7:
                    days_left   = max(1, c.end_day - today + 1)
                    daily_limit = remB / days_left
                    bid_entries = set()
                    # weight across segments by expected supply (SEG_FREQ)
                    for seg, freq in self.SEG_FREQ.items():
                        if c.target_segment.issubset(seg):
                            # fraction of supply in seg
                            total_freq = sum(self.SEG_FREQ[s] for s in self.SEG_FREQ
                                            if c.target_segment.issubset(s))
                            w = freq / total_freq
                            Ls = daily_limit * w
                            # pacing bid capped by marginal value
                            rho_x  = self.effective_reach(done,   c.reach)
                            rho_x1 = self.effective_reach(done+1, c.reach)
                            mv   = (rho_x1 - rho_x) * c.budget
                            ps = min(max(0.1, Ls / (remR * w)),mv)
                            bid_entries.add(Bid(self, seg, bid_per_item=ps,
                                            bid_limit=Ls))
                    bundles.add(BidBundle(c.uid, limit=daily_limit,
                                        bid_entries=bid_entries))
                    continue
                else:
                    continue

        return bundles

    def estimate_campaign_difficulty(self, campaign: Campaign) -> float:
        '''
        Method to estimate campaign difficulty score with value ~ [0,1]. 
        This score will be used to scale bids used to auction for a campaign.
        A campaign is harder if more other campaigns compete for the same users.
        '''
        active = list(self.get_active_campaigns())
        # no difficulty if it is the only campaign
        if len(active) <= 1:
            return 0.0
        overlap = 0
        # count how many other campaigns share any subsegment
        for ac in active:
            if ac is campaign:
                continue
            if ac.target_segment & campaign.target_segment:
                overlap += 1
        # normalize by number of other campaigns
        return overlap / (len(active) - 1)


    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        # bids = {}
        # # today = self.get_current_day()

        # # for c in campaigns_for_auction:
        # #     # PHASE 1: days 1–7, only bid on campaigns that end by day 7
        # #     if today <= 7:
        # #         if c.end_day <= 7:
        # #             # bid at the maximum (reach R) to guarantee you win
        # #             bids[c] = 0.1*c.reach
        # #     # PHASE 2: days 8–10, revert to equilibrium bids for campaigns ending after 7
        # #     else:
        # #         if c.end_day > 7:
        # #             bids[c] = self._best_campaign_bid(c.reach)
        # # return bids
        # today = self.get_current_day()
        # for c in campaigns_for_auction:
        #     # DAYS 1–7: only short campaigns, bid minimum to win cheaply
        #     # if today <= 7 and c.end_day <= 7:
        #     #     bids[c] = 0.1 * c.reach
        #     # # DAYS 8–10: profit phase, bid Bayes‐optimal
        #     # elif today > 7 and c.end_day > 7:
        #     #     seg = c.target_segment
        #     #     cdf = self.prior_cdf.get(seg)
        #     #     pdf = self.prior_pdf.get(seg)
        #     #     if cdf and pdf:
        #     #         # print(c.reach)
        #     #         bids[c] = self.optimize_b_reverse_2nd_price(R=c.reach, cdf=cdf,pdf= pdf, m=10)
        #     seg = c.target_segment
        #     cdf = self.prior_cdf.get(seg)
        #     pdf = self.prior_pdf.get(seg)
        #     if cdf and pdf:
        #         # print(c.reach)
        #         bids[c] = self.optimize_b_reverse_2nd_price(R=c.reach, cdf=cdf,pdf= pdf, m=10)

        # Max's implementation of campaign difficulty bidding
        bids = {}
        for c in campaigns_for_auction:
            diff = self.estimate_campaign_difficulty(c)
            raw  = diff * c.reach
            bid  = self.clip_campaign_bid(c, raw)
            bids[c] = bid
        return bids

    
    @staticmethod
    def num_agents() -> int:
        # total number of competitors (including us)
        return 10

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)