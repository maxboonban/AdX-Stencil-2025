from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent 
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent 
from agt_server.local_games.adx_arena import AdXGameSimulator 
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
from typing import Set, Dict
import math

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = "hmax_naive"  # TODO: enter a name.
        self.a = 4.08577
        self.b = 3.08577
        self.eta_low = 0.9
        self.eta_high = 1.3


    def on_new_game(self) -> None:
        # TODO: fill this in (if necessary)
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        # TODO: fill this in
        bundles = set()

        # Iterate over all active campaigns
        for c in self.get_active_campaigns():
            spent = self.get_cumulative_cost(c)
            remaining = max(0.0, c.budget - spent)
            bids = set()
            
            segment_users = {
                frozenset(['Male', 'Young', 'LowIncome']): 1836,
                frozenset(['Male', 'Young', 'HighIncome']): 517,
                frozenset(['Male', 'Old', 'LowIncome']): 1795,
                frozenset(['Male', 'Old', 'HighIncome']): 808,
                frozenset(['Female', 'Young', 'LowIncome']): 1980,
                frozenset(['Female', 'Young', 'HighIncome']): 256,
                frozenset(['Female', 'Old', 'LowIncome']): 2401,
                frozenset(['Female', 'Old', 'HighIncome']): 407
            }

            # compute optimal reach fraction pacing
            x = self.get_cumulative_reach(c)
            R = c.reach
            B = c.budget
            # estimate cost per impression
            k = (spent / x) if x > 0 else (B / R)
            # solve for raw eta: eta = (b + sqrt(max(0, 2B/(kR)-1)))/a
            raw = 0.0
            term = 2 * B / (k * R) - 1
            if term > 0:
                raw = self.b + math.sqrt(term)
            eta = raw / self.a
            # clip eta to [eta_low, eta_high]
            eta = min(self.eta_high, max(self.eta_low, eta))
            # pace daily target based on eta
            duration = c.end_day - c.start_day
            # daily_target = c.reach / duration
            daily_target = eta * R / duration
            # print(daily_target)
            
            for seg in MarketSegment.all_segments():
                # print(seg)
                # find matching segment by checking if seg is subset of any segment_users key
                matching_segment = None
                for user_segment in segment_users:
                    if seg.issubset(user_segment):
                        matching_segment = user_segment
                        break
                        
                if matching_segment:
                    # scale bid based on segment size relative to campaign needs
                    # higher bid for segments closer to our daily target
                    segment_size = segment_users[matching_segment]
                    # print(segment_size)
                    #print(f"{type(daily_target)} daily target and {type(segment_size)} is the segment size")
                    bid_per_item = min(10.0, (daily_target / segment_size)*5.0)
                    
                    # add bid with calculated bid_per_item
                    bids.add(Bid(
                        bidder=self,
                        auction_item=seg,
                        bid_per_item=bid_per_item,
                        bid_limit=remaining
                    ))

            # Iterate over all market segments. Add a bid of $1 to every market segment
            # for seg in MarketSegment.all_segments():
            #     bids.add(Bid(
            #         bidder=self,
            #         auction_item=seg,
            #         bid_per_item=1.0,
            #         bid_limit=remaining
            #     ))

            bundles.add(BidBundle(
                campaign_id=c.uid,
                limit=remaining,
                bid_entries=bids
            ))

        return bundles

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
        # TODO: fill this in
        '''
        High level strategy: Since it's a second-price auction, truthful bidding is a DSIC. 
        We will shade off 80% of campaign bid as a strategy.
        '''
        bids = {}
        for c in campaigns_for_auction:
            # only bid 80% of campaign's reach
            adjusted_budget = 0.8 * c.reach
            bid = self.clip_campaign_bid(c, adjusted_budget)
            bids[c] = bid

        return bids

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)#num_simulations=500)