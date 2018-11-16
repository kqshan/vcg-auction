#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vickrey-Clarke-Groves (VCG) auction.

Kevin Shan, 2018-11-08
"""
import re
from collections import Counter
from collections import namedtuple
from collections import defaultdict
import numpy as np
import datetime
import argparse

class Bidder:
    """A class indicating the bidders and their bids
    
    Attributes:
        name        Name of the bidder
        manual_bids List of manually-entered (items,price) bids
    """
    
    Bid = namedtuple('Bid',['items','price'])
    
    def __init__(self, name=""):
        self.name = name
        self.manual_bids = list()
    
    def add_bid(self, items, bid):
        """Add a new bid to this bidder
        
        Args:
            items   List of item names indicating the combination being bid on
            bid     Bid price (float)
        """
        self.manual_bids.append(self.Bid(items,bid))
        
    def __str__(self):
        header = ''
        if self.name:
            header += "Name: {}\n".format(self.name)
        header += "Manual bids:\n"
        header += "   Price  Items\n"
        lines = ['{:>8}  {}'.format(x.price,' & '.join(x.items))
                 for x in self.manual_bids]
        return header + '\n'.join(lines)

    
class Auction:
    """A class indicating the items for auction
    
    Attributes:
        item_names  Dict mapping item names (lowercase) to original names
        item_counts Dict mapping item names (lowercase) to quantities (int)
    """
    
    def __init__(self):
        self.item_names = dict()
        self.item_counts = Counter()
    
    def add_item(self, name, qty=1):
        """Add an item to this auction
        
        Args:
            name    Item name. Case insensitive for matching, but the
                    capitalization will be preserved for display.
            qty     Quantity of this item to add. Default=1
        """
        name = name.strip()
        key_name = name.lower()
        if key_name not in self.item_names:
            self.item_names[key_name] = name
        self.item_counts[key_name] += qty
    
    def split_auction(self, bidders):
        """Split this auction into sub-auctions based on item overlap in bids
        
        Args:
            bidders     List of Bidder objects
        Returns:
            subauctions List of (Auction obj, list of Bidder objs)
        """
        # Start by creating a separate sub-auction for each unique item
        Subauction = namedtuple('Subauction',['auction','bidders'])
        item_sa_map = {item:Subauction({item},defaultdict(list))
                       for item in self.item_names}
        # Merge subauctions if a bid includes items from multiple subauctions
        for bidder in bidders:
            for bid in bidder.manual_bids:
                # Find the subauction that includes these items
                last_sa = None
                for item in bid.items:
                    curr_sa = item_sa_map[item]
                    if (last_sa) and (curr_sa is not last_sa):
                        # Merge last_sa into curr_sa
                        curr_sa.auction.update(last_sa.auction)
                        for name,bids in last_sa.bidders.items():
                            curr_sa.bidders[name].extend(bids)
                        for item in last_sa.auction:
                            item_sa_map[item] = curr_sa
                    last_sa = curr_sa
                # Add this bid to that subauction
                curr_sa.bidders[bidder.name].append(bid)
        # Get the unique subauctions
        subauctions = {id(sa):sa for sa in item_sa_map.values()}
        subauctions = list(subauctions.values())
        # Convert to the desired objects
        for idx,sa in enumerate(subauctions):
            # Convert the auction from a list of items to an Auction obj
            auction = Auction()
            for item in sa.auction:
                auction.add_item(self.item_names[item], self.item_counts[item])
            # Construct the Bidder list
            bidders = list()
            for name,bids in sa.bidders.items():
                bidder = Bidder(name)
                for bid in bids:
                    bidder.add_bid(bid.items, bid.price)
                bidders.append(bidder)
            # Add this tuple to the subauction
            subauctions[idx] = Subauction(auction, bidders)
        return subauctions
    
    def __contains__(self, item):
        return item in self.item_names
    
    def __str__(self):
        header = 'Qty  Name\n';
        lines = ['{:>3}  {}'.format(self.item_counts[x],self.item_names[x]) 
                 for x in self.item_names]
        return header + '\n'.join(lines)



class AuctionProblem:
    """Auction problem that will be solved using an exhaustive search
    
    Attributes:
        items       List of item names (lowercase)
        bidders     List of bidder names
        item_counts Item counts (same order as in "items")
        values      [B x N1+1 x N2+1 x ... x Nm+1] valuations for all combos
                    of the m unique items by each of the B bidders
    """
    
    def __init__(self, auction, bidders):
        """Construct a new AuctionProblem from an auction and some bidders
        
        Args:
            auction     Auction object
            bidders     List of Bidder objects
        
        This populates self.items, .bidders, and .values
        """
        # Decide on the canonical order of items and bidders
        items = auction.item_counts.most_common()
        items = [x[0] for x in items]
        bidder_names = [bidder.name for bidder in bidders]
        # Make sure this isn't going to be too large
        item_counts = [auction.item_counts[item] for item in items]
        item_counts = np.array(item_counts, dtype=np.intp)
        nBidders = len(bidders)
        dims = item_counts + 1
        nCases = np.prod(dims)
        if nCases*nBidders > 100e6:
            raise ValueError("Problem size {} is too large; aborting"
                             .format(np.append(dims,nBidders)))
        # Some helpers for the indexing
        nItems = len(items)
        item_idx = {item:items.index(item) for item in items}
        # Create a [N1+1 x N2+1 x ... x Nm+1] array for each bidder
        values = np.zeros(np.hstack((nBidders,dims)), dtype=np.float32)
        for bidder in bidders:
            # Populate the value array for this bidder
            v = np.zeros(dims, dtype=np.float32)
            for bid in bidder.manual_bids:
                # Count how many of each item is in this bid
                idx_list = [item_idx[item] for item in bid.items]
                bid_counts = np.bincount(idx_list, minlength=nItems)
                if any(bid_counts > item_counts):
                    print("{}'s bid of {} has too many items; ignoring"
                          .format(bidder.name, bidder.items))
                    continue
                # value with items = value without items + bid price
                endpts = item_counts - bid_counts + 1
                src_slice = tuple(slice(None,n) for n in endpts)
                tgt_slice = tuple(slice(n,None) for n in bid_counts)
                v[tgt_slice] = np.maximum(v[tgt_slice], v[src_slice]+bid.price)
            # Save it
            b = bidder_names.index(bidder.name)
            values[b,...] = v
        # Update this object
        self.items = items
        self.bidders = bidder_names
        self.values = values
        self.item_counts = item_counts
        # And some private fields
        dims_right = np.append(dims[1:],1)
        self._value_array_strides = np.cumprod(dims_right[::-1])[::-1]
        self._bidder_strides = np.array(range(nBidders)) * nCases
    
    def get_value(self, alloc):
        """Return the value of a given Allocation object
        """
        # alloc.counts is a [#bidders x #items] matrix, and
        # self._value_array_strides is a [#items] vector indicating the stride
        # size of each dimension (assuming row-major ordering)
        case_indices = np.matmul(alloc.counts, self._value_array_strides)
        # Now add the offsets for each of the bidders
        case_indices += self._bidder_strides
        # Look these up in the value array and sum the result
        return np.sum(self.values.flat[case_indices])
    
    def __str__(self):
        lines = []
        # This table header gets used in both cases
        N_a = self.values.shape[1]
        table_header = ['{:_>7d}'.format(i) for i in range(N_a)]
        table_header = '_'.join(table_header) + ' \u2190 ' + self.items[0]
        # Special case if there's only one item
        if len(self.items)==1:
            width = max([len(x) for x in self.bidders])
            lines.append(' '*width + '  ' + table_header)
            for b,name in enumerate(self.bidders):
                bidder = '{:>{width}s}'.format(name, width=width)
                vals = ['{:7g}'.format(v) for v in self.values[b,:]]
                lines.append(bidder + ': ' + ' '.join(vals))
            return '\n'.join(lines)
        # Print each bidder in their own section
        for b,name in enumerate(self.bidders):
            lines.append("{}:".format(name))
            v_bidder = self.values[b,...]
            # Flatten this to 3 dimensions
            item_b = self.items[1]
            items_c = self.items[2:]
            N_b = v_bidder.shape[1]
            shape_c = v_bidder.shape[2:]
            nPages = np.prod(shape_c, dtype=np.int)
            v_bidder.shape = (N_a, N_b, nPages)
            # Produce 2-D tables in a loop over the 3rd dimension
            for p in range(nPages):
                # Display the page info
                indent = '    '
                if (nPages > 1):
                    idx = np.unravel_index(p, shape_c)
                    line = ["{} {}".format(*x) for x in zip(idx,items_c)]
                    lines.append(indent + ', '.join(line))
                    indent += '    '
                v_page = v_bidder[:,:,p]
                # Display the header
                lines.append(indent + '\u256d\u2500 ' + item_b)
                lines.append(indent + '\u2193 ' + table_header)
                # Display the values
                for b in range(N_b):
                    vals = ['{:7g}'.format(v) for v in v_page[:,b]]
                    lines.append(indent + '{}|'.format(b) + ' '.join(vals))
            # Add a separator between bidders
            lines.append('-'*80)
        return '\n'.join(lines)



class Allocation:
    """Class indicating an allocation of items to bidders
    
    Attributes:
        counts      [#bidders x #items] counts of items allocated to bidders
    """
    
    def __init__(self, counts):
        self.counts = np.array(counts, dtype=np.int)

    def to_hashable(self):
        """Return a hashable value from this object"""
        return self.counts.tostring()
    
    def emptyhanded(self):
        """Return an iterator over bidders that were assigned no items"""
        is_emptyhanded = (np.sum(self.counts,1) == 0)
        return np.nditer(np.where(is_emptyhanded),('zerosize_ok',))
    
    def give_item(self, item_idx, bidder_idx):
        """Create a new Allocation object with an additional item allocated
        
        Args:
            item_idx    Index of the item to allocate
            bidder_idx  Index of the bidder to allocate it to
        """
        new_counts = np.copy(self.counts)
        new_counts[bidder_idx,item_idx] += 1
        return Allocation(new_counts)

    

class AuctionSolver:
    """Class to exhaustively explore all item allocations in an auction
    
    Attributes:
        problem     AuctionProblem object
        seen_allocs Set of allocations that have been considered so far
        best_alloc  List of allocations that can achieve best_val
        best_val    Best overall value seen so far
        best_val_wo [#bidders] array of best overall value without each bidder
    """
    def __init__(self, problem, verbose=True):
        # Define the public attributes
        self.problem = problem
        self.seen_allocs = set()
        self.best_val = -np.inf
        self.best_alloc = []
        nBidders = len(problem.bidders)
        nItems = len(problem.items)
        self.best_val_wo = np.full(nBidders, -np.inf, dtype=np.float32)
        # Define some private attribets
        self._nBidders = nBidders
        self._nItems = nItems
        self._item_counts = problem.item_counts
        self._verbose = verbose
        # Perform the search
        if self._verbose:
            print("{}: Started search".format(datetime.datetime.now()))
        starting_alloc = Allocation(np.zeros((nBidders,nItems), dtype=np.int))
        self.explore(starting_alloc)
        if self._verbose:
            print("{}: Finished search (checked {} cases)"
                  .format(datetime.datetime.now(), len(self.seen_allocs)) )
    
    def explore(self, alloc):
        """Recursive function to perform a depth-first search over allocations
        
        Args:
            alloc   Allocation object to start from
        """
        # Mark this allocation as seen and compare it to the others so  far
        self.evaluate(alloc)
        # Search the rest
        items_allocated = np.sum(alloc.counts,0)
        for b in range(self._nBidders):
            for i in range(self._nItems):
                # Stop if we can't add any more of that item
                if items_allocated[i] == self._item_counts[i]:
                    continue
                new_alloc = alloc.give_item(i,b)
                # Recurse if this has not yet been seen
                if new_alloc.to_hashable() not in self.seen_allocs:
                    self.explore(new_alloc)

    def evaluate(self, alloc):
        """Evaluate a given Allocation object, adding it to our running logs of
        which allocations we've seen and the best so far"""
        self.seen_allocs.add(alloc.to_hashable())
        # Check if this is the best so far
        val = self.problem.get_value(alloc)
        if val > self.best_val:
            self.best_val = val
            self.best_alloc = [alloc]
        elif val == self.best_val:
            self.best_alloc.append(alloc)
        # Check if this is the best among allocations that exclude bidders
        for b in alloc.emptyhanded():
            if val > self.best_val_wo[b]:
                self.best_val_wo[b] = val
        # Print progress if desired
        if self._verbose:
            n = len(self.seen_allocs)
            if n % 10000 == 0:
                print("{}: Checked {} possible cases so far"
                      .format(datetime.datetime.now(),n) )


def parse_auction_specs(file):
    """Read the auction specification from a file
    Args:
        file        File object that we will read line-by-line
    Returns:
        auction     Auction object with items for sale
        bidders     List of Bidder objects for each bidder
    """
    # Compile some regular expressions
    num_str_re = re.compile(
            '^\s*(?P<num>(\d+(\.\d*)?)|(\.\d+))?\s*(?P<name>[a-zA-Z].*?)\s*$')
    midx_num = num_str_re.groupindex['num']
    midx_name = num_str_re.groupindex['name']
    key_val_re = re.compile(
            '^\s*(?P<key>\w[\w\s]*?)\s*:\s*(?P<val>\w[\w\s.]*?)\s*$')
    midx_key = key_val_re.groupindex['key']
    midx_val = key_val_re.groupindex['val']
    # Initialize the loop
    parser_state = 'expecting item list'
    auction = Auction()
    bidders = []
    this_bidder = Bidder("Bidder #{}".format(len(bidders)+1))
    # Read each line of the file
    lineNo = 0;
    for line in file:
        # Generate a general error message that we might use
        lineNo += 1
        errmsg = 'Could not parse line {}'.format(lineNo)
        errmsg += ' "{}"'.format(line.rstrip('\n')) + '; expected {}'
        # Put the line into our state machine
        if (line.isspace() or line.startswith('#')):
            # Ignore empty lines and comments
            pass
        elif parser_state == 'expecting item list':
            # The first nonempty line should be "items to sell:"
            if line.lower().startswith('items to sell'):
                parser_state = 'item list'
            else:
                raise InputParseError(errmsg.format('"Items to sell:"'))
        elif parser_state == 'item list':
            # Now we are expecting item names and possibly quantities
            if line.startswith('---'):
                parser_state = 'bidder header'
                continue
            # Parse the line
            m = num_str_re.match(line)
            if m is None:
                raise InputParseError(errmsg.format("item name"))
            name = m.group(midx_name)
            qty = int(m.group(midx_num)) if m.group(midx_num) else 1
            auction.add_item(name, qty)
        elif parser_state == 'bidder header':
            # Fill in the properties of the bidder
            if line.lower().startswith('bids'):
                parser_state = 'bids'
                continue
            # Line should be "attribute: value" pair
            m = key_val_re.match(line)
            if m is None:
                raise InputParseError(errmsg.format(
                        'something like "Bidder attribute: value"'))
            attr = m.group(midx_key).lower()
            val = m.group(midx_val)
            if attr.startswith('name'):
                this_bidder.name = val
            else:
                raise InputParseError(errmsg.format("a valid attribute name"))
        elif parser_state == 'bids':
            # Now we are expecting individual bids
            if line.startswith('---'):
                # We're now done with this bidder; start a new one
                bidders.append(this_bidder)
                this_bidder = Bidder("Bidder #{}".format(len(bidders)+1))
                parser_state = 'bidder header'
                continue
            # Parse the line
            m = num_str_re.match(line)
            if m is None:
                raise InputParseError(errmsg.format("bid and item name"))
            bid = m.group(midx_num)
            if bid is None:
                raise InputParseError(errmsg.format("bid price"))
            bid = float(bid)
            items = m.group(midx_name).split('&')
            items = [item.strip().lower() for item in items]
            # Check that they are valid
            bad_items = [x for x in items if x not in auction]
            if bad_items:
                print('Bidder "{}": '.format(this_bidder.name) +
                      'Your bid "{}" was ignored '.format(line.rstrip('\n')) + 
                      'because "{}" is not an item name'.format(bad_items[0]))
                continue
            # Record the bid
            this_bidder.add_bid(items, bid)
        else:
            raise AssertionError('Unknown state "{}"'.format(parser_state))
        # Go on to the next line in the file
    # Reached end of file
    if parser_state == 'bidder header': # Everything is good
        if not bidders:
            raise InputParseError("No bidders found")
    elif parser_state == 'bids':        # Forgat a closing "---"; that's okay
        bidders.append(this_bidder)
    else:
        raise InputParseError("Unexpected end of file")
    # Check that the bidder names are unique
    bidder_name_counts = Counter([b.name for b in bidders])
    [most_common_bidder] = bidder_name_counts.most_common(1)
    if (most_common_bidder[1] > 1):
        raise InputParseError('Duplicate bidder "{}" found'
                              .format(most_common_bidder[0]))
    # Return the auction and bidders
    return auction, bidders

     

class InputParseError(ValueError):
    """An exception class indicating that we could not parse the input text
    """
    pass



def run_auction(file, verbose=0):
    """Run an auction given a file
    """
    more_verbose = (verbose >= 2)
    # Parse the input
    auction,bidders = parse_auction_specs(file)
    if verbose:
        print("Found {} unique items and {} bidders"
              .format(len(auction.item_names), len(bidders)) )
    if more_verbose:
        print(auction)
        for b in bidders:
            print('-'*10)
            print(b)
        print('='*80)
    # Split this into sub-auctions
    subauctions = auction.split_auction(bidders)
    if verbose:
        print("Able to split this into {} independent sub-auctions"
              .format(len(subauctions)) )
    problems = [AuctionProblem(x.auction,x.bidders) for x in subauctions]
    # Solve the sub-auctions
    for i,p in enumerate(problems):
        if verbose:
            print("Problem {}:".format(i+1))
        solution = AuctionSolver(p, verbose=more_verbose)
        if verbose:
            print("I solved it I swear")



if __name__ == '__main__':
    """Entry point if running this module from the shell"""
    parser = argparse.ArgumentParser(description="Parse an input file " +
        "with items/bids and print the optimal allocation and prices")
    parser.add_argument('infile', help="Input file " + 
        "(see example_input.txt for an explanation of the format)")
    parser.add_argument('-v','--verbose', action='count', default=0,
        help="Enable verbose output (more v's = more verbosity)")
    args = parser.parse_args()
    # Call the main function
    file = open(args.infile,'r')
    run_auction(file, verbose=args.verbose)
