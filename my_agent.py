from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
from typing import Set, Dict
import numpy as np

from my_agent_smart_prev import MyNDaysNCampaignsAgent as chillguytest
from exp_agent import MyNDaysNCampaignsAgent as exp_agent
chillguytest = chillguytest()
exp_agent = exp_agent()

SEG_FREQ = {
    MarketSegment({'Male','Young','LowIncome'}): 1836,
    MarketSegment({'Male','Young','HighIncome'}):  517,
    MarketSegment({'Male','Old','LowIncome'}): 1795,
    MarketSegment({'Male','Old','HighIncome'}):  808,
    MarketSegment({'Female','Young','LowIncome'}): 1980,
    MarketSegment({'Female','Young','HighIncome'}):  256,
    MarketSegment({'Female','Old','LowIncome'}): 2401,
    MarketSegment({'Female','Old','HighIncome'}):  407,
}

ALL_SEG_FREQ = {
    MarketSegment({'Male',   'Young',    'LowIncome'}): 1836,
    MarketSegment({'Male',   'Young',    'HighIncome'}):  517,
    MarketSegment({'Male',   'Old',      'LowIncome'}): 1795,
    MarketSegment({'Male',   'Old',      'HighIncome'}):  808,
    MarketSegment({'Female', 'Young',    'LowIncome'}): 1980,
    MarketSegment({'Female', 'Young',    'HighIncome'}):  256,
    MarketSegment({'Female', 'Old',      'LowIncome'}): 2401,
    MarketSegment({'Female', 'Old',      'HighIncome'}):  407,
    MarketSegment({'Male',   'Young'}): 2353,
    MarketSegment({'Male',   'Old'}):   2603,
    MarketSegment({'Female', 'Young'}): 2236,
    MarketSegment({'Female', 'Old'}):   2808,
    MarketSegment({'Male',   'LowIncome'}):  3631,
    MarketSegment({'Male',   'HighIncome'}): 1325,
    MarketSegment({'Female', 'LowIncome'}):  4381,
    MarketSegment({'Female', 'HighIncome'}):  663,
    MarketSegment({'Young', 'LowIncome'}): 3816,
    MarketSegment({'Young', 'HighIncome'}): 773,
    MarketSegment({'Old',   'LowIncome'}): 4196,
    MarketSegment({'Old',   'HighIncome'}):1215,
    MarketSegment({'Male'}):   4956,
    MarketSegment({'Female'}): 5044,
    MarketSegment({'Young'}):  4589,
    MarketSegment({'Old'}):    5411,
    MarketSegment({'LowIncome'}):  8012,
    MarketSegment({'HighIncome'}): 1988,
}
# Tunable initial competitiveness factor
INITIAL_COMPETITIVENESS = 0.5

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = "chillguy"  # TODO: enter a name.
        #storing a list of campaigns currently ongoing 
        self.known_campaigns: list[(int, Campaign)] = []
        self.competitiveness = INITIAL_COMPETITIVENESS

    def on_new_game(self) -> None:
        # TODO: fill this in (if necessary)
        self.competitiveness = INITIAL_COMPETITIVENESS
        self.known_campaigns.clear()
    
    def _cleanup_known(self, today: int):
        self.known_campaigns = [ (d,c) for (d,c) in self.known_campaigns if d >= today ]
    
    def _add_known(self, campaign: Campaign):
        self.known_campaigns.append((campaign.end_day, campaign))
    
    def _segment_cov(self, S: MarketSegment, t: int) -> float:
        # known-campaign coverage heuristic
        cov = 0.0
        tot_cam = 0
        for end_d, c in self.known_campaigns:
            if c.start_day <= t <= end_d and c.target_segment.issubset(S):#S.issubset(c.target_segment):
                tot_cam +=1
                # cov += c.reach * (len(S)/len(c.target_segment))
                tot_dur = end_d - c.start_day +1
                cov += c.reach *(t-c.start_day+1)/tot_dur
        #print(f"pk is {cov/10000}")
        return cov/10000
    
    def _expected_segment_cov(self, S: MarketSegment, t: int) -> float:
        # competition from unknown campaigns -> use expectations
        Qavg = self.get_quality_score()
        delta_avg = (0.3 + 0.5 + 0.7)/3
        p_S = ALL_SEG_FREQ[S]/10000  # uniform over 20 segments -> because all auctioned campaingns are combinations of 2 or 3 features 
        #print(f"pU is {Qavg * delta_avg * p_S * 2}")
        return Qavg * delta_avg * p_S * 2
    
    def _difficulty(self, c: Campaign) -> float:
        # avg of pK+pU over campaign days, divided by cube-root length
        l = c.end_day - c.start_day + 1
        total = 0.0
        for t in range(c.start_day, c.end_day+1):
            total += self._segment_cov(c.target_segment, t) + self._expected_segment_cov(c.target_segment, t)
            # print(total)
        avg = total
        return avg/l 
    

    def get_ad_bids(self) -> Set[BidBundle]:
        # TODO: fill this in
        bundles = set()
        today = self.get_current_day()
        print("*******************************")
        print(f"number of active campaigns on Day {today} is {len(self.get_active_campaigns())}. My quality score is {self.get_quality_score()}")
        for i,c in enumerate(self.get_active_campaigns()):
            print(f"Campaign {i}: {c.budget}, {c.reach}, {c.target_segment}, startday {c.start_day}, endDay {c.end_day}")
        print(f"current profit {self.profit}")
        for c in self.get_active_campaigns():
            spent = self.get_cumulative_cost(c)
            done  = self.get_cumulative_reach(c)
            remB  = max(0.0, c.budget - spent) #remaining budget -> for making profit
            remR  = max(1,     c.reach  - done) # remaining reach to be fulfilledx

            # compute clipped target reach fraction eta star
            B, R = c.budget, c.reach
            # k is the cost (our valuation) of a single impression. If we satisfy the campaign set this to a high value.
            k = remB/remR if remR else B
            eta = 3.08577 + np.sqrt(max(0, 2*B/(k*R) - 1)) / 4.08577
            eta_star = min(1.3, max(0.9, eta))

            # daily limit
            days_left = max(1, c.end_day - today + 1)
            if today <= 7 and c.end_day <= 7:
                # lets try to finish the campaigns -> to increase quality score
                dailyL = remB
                Bhat = 0.55 * (c.budget - spent)
            else:
                # here the aim is to make profit
                dailyL = remB / days_left
                Bhat = 0.51 * (c.budget - spent)
            # last-day urgency
            if today == c.end_day and (done / c.reach) < eta_star:
                dailyL *= 2

            # Bhat because there is no incentive in reporting true budget
            # Bhat = 0.51 * (c.budget - spent)
            if done/R >= eta_star:
                Bhat = 0.01 * (c.budget - spent)
            bid_per_item = Bhat / remR

            # segment-weighted bids
            bid_entries = set()
            total_f = sum(v for s,v in SEG_FREQ.items() if c.target_segment.issubset(s))
            for s,v in SEG_FREQ.items():
                if c.target_segment.issubset(s):
                    w = v / total_f
                    Ls = dailyL * w
                    bid_entries.add(Bid(self, s, bid_per_item=bid_per_item, bid_limit=Ls))

            bundles.add(BidBundle(c.uid, limit=dailyL, bid_entries=bid_entries))
        return bundles

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
        # TODO: fill this in 
        bids = {}
        today = self.get_current_day()
        # remove the campaigns that end today from the known campaign list
        self._cleanup_known(today)
        # TODO: add today's free campaign
        for c in campaigns_for_auction:
            if today <= 1:
                # quality phase: win cheaply longer campaigns so as to increase quality score by satifying them
                bids[c] = 0.1 * c.reach
                self._add_known(c)
            elif today > 1:
                # profit phase: difficulty-based bid
                diff = self._difficulty(c)
                raw = self.competitiveness * diff*2
                #print(f"difficulty {diff}, raw {raw}, reach {c.reach}")
                bids[c] = max(0.1*c.reach, min(raw, c.reach))
                self._add_known(c)
        return bids

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)] 

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=5)


# def score_campaign(self,c):
#         R = c.reach
#         base = { s: SEG_FREQ[s]/10000.0    # or some constant baseline, e.g. 0.001
#             for s in SEG_FREQ }
#         # 1. estimate per-impression cost from segment baselines
#         avg_reserve = sum(base[s] for s in SEG_FREQ if s.issuperset(c.target_segment))
#         freq_sum    = sum(SEG_FREQ[s]     for s in SEG_FREQ if s.issuperset(c.target_segment))
#         p_hat       = avg_reserve / freq_sum

#         # 2. compute η* for profit
#         B = R
#         eta = 3.08577 + np.sqrt(max(0, 2*B/(1*R) - 1)) / 4.08577
#         eta_star = min(1.3, max(0.9, eta))


#         # 3. gross surplus
#         surplus = B * self.effective_reach(eta_star * R, R) - p_hat * eta_star * R

#         # 4. difficulty
#         days_left = c.end_day - self.get_current_day() + 1
#         remR      = R - self.get_cumulative_reach(c)
#         remB_est  = p_hat * remR
#         d = (p_hat * remR) / max(remB_est / days_left, 1e-3) * np.sqrt(days_left)

#         # 5. quality bonus
#         Q_bonus = 1 + 0.5*(1 - self.get_quality_score()) if c.end_day <= 7 else 1.0

#         # 6. difficulty discount
#         D_disc = np.exp(-0.5 * d)

#         V = surplus * Q_bonus * D_disc
#         return np.clip(V, 0.1*R, R)

#     def get_campaign_bids(self,campaigns):
#         bids = {}
#         for c in campaigns:
#             v = self.score_campaign(c)
#             bids[c] = v
#         return bids