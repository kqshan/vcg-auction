# vcg-auction
Python code for implementing a Vickrey-Clarke-Groves (VCG) auction

# What does this do?
A VCG auction is a form of sealed-bid auction in which bidders may bid on multiple items and combinations thereof. In theory, it is socially optimal and incentivizes truthful bidding: each bidder will bid their true valuations for the items.

# How do I use this?
Look at `example_input.txt` for the input specification. Following that example, you create a text file that lays out what the items for auction are, and each bidder's bids.

Then run the Python code. If you are running Linux, this is super easy. Change to the directory and:

    $ ./vcg_auction.py example_input.txt

If you want to see more details, add `-v` or even `-vv` for more verbose output:

    $ ./vcg_auction.py -vv example_input.txt

This will also work (and maybe even work on Windows/OSX?):

    $ python3 vcg_auction.py example_input.txt

If you are using Spyder or IDLE or whatever else, I think this works:

    >>> import vcg_auction
    >>> vcg_auction.run_auction(open('example_input.txt'), verbose=0)

# What's with the example items/bidders?

This was inspired by a loot spliting problem in a D&D campaign.

# Limitations

## Bids are assumed to be supermodular

There are some stupid restrictions on what kind of bids are allowed. In order to make life easier for the human bidder, we try to infer what you would bid on combinations of items that you didn't explicitly bid on.

So if you bid $1 on A and $2 on B, we infer that you would be willing to pay at least $3 for the combination of A & B. If you explicitly bid $5 on the combination A & B, then that would supersede the inferred bid of $3. However, if you bid less than $3 on the combination A & B, then we kinda just ignore that because that makes life more difficult for us to figure out what you want.

## Complexity gets pretty bad pretty fast

The complexity grows exponentially with the number of bidders and the maximum number of items that are included in a single combination bid (i.e. a bid that includes multiple items). So easiest way to keep things managable is to discourage bidders from placing large combination bids.

From a computational point of view, there are a few modifications that could probably help things out a bit:
* Removing dominated bidders before the exhaustive search
* Early stopping when we reach the point where a bidder has zero marginal utility for additional quantities of a particular item
* Somehow taking advantage of the supermodularity so that we don't have to preform an exhaustive search
