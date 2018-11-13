# -*- coding: utf-8 -*-
"""
Vickrey-Clarke-Groves (VCG) auction.

Kevin Shan, 2018-11-08
"""
import re
import math

            
class Bidder:
    """A class indicating the bidders and their bids
    
    Attributes:
        name        Name of the bidder
        default_bid Amount to implicitly bid on all items
        budget      Maximum allowed to spend on all winning bids
        manual_bids List of manually-entered (item_list,bid_price)
    """
    
    def __init__(self):
        self.name = ''
        self.default_bid = 0.0
        self.budget = math.inf
        self.manual_bids = list()
    
    def add_bid(self, items, bid):
        """Add a new bid to this bidder
        
        Args:
            items   List of item names indicating the combination being bid on
            bid     Bid price (float)
        """
        self.manual_bids.append((items,bid))
    
    def __str__(self):
        header = ''
        if self.name:
            header += "Name: {}\n".format(self.name)
        header += "Default bid: {}\n".format(self.default_bid)
        header += "Budget: {}\n".format(self.budget)
        header += "Manual bids:\n"
        header += "   Price  Items\n"
        lines = ['{:>8}  {}'.format(x[1],' & '.join(x[0]))
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
        self.item_counts = dict()
    
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
            self.item_counts[key_name] = 0
        self.item_counts[key_name] += qty
    
    def __contains__(self, item):
        return item in self.item_names
    
    def __str__(self):
        header = 'Qty  Name\n';
        lines = ['{:>3}  {}'.format(self.item_counts[x],self.item_names[x]) 
                 for x in self.item_names]
        return header + '\n'.join(lines)



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
    this_bidder = Bidder()
    this_bidder.name = "Bidder #{}".format(len(bidders)+1)
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
            elif attr.startswith('budget'):
                this_bidder.budget = float(val)
            elif attr.startswith('default bid'):
                this_bidder.default_bid = float(val)
            else:
                raise InputParseError(errmsg.format("a valid attribute name"))
        elif parser_state == 'bids':
            # Now we are expecting individual bids
            if line.startswith('---'):
                # We're now done with this bidder; start a new one
                bidders.append(this_bidder)
                this_bidder = Bidder()
                this_bidder.name = "Bidder #{}".format(len(bidders)+1)
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
        pass
    elif parser_state == 'bids':        # Forgat a closing "---"; that's okay
        bidders.append(this_bidder)
    else:
        raise InputParseError("Unexpected end of file")
    # Return the auction and bidders
    return auction, bidders

     

class InputParseError(ValueError):
    """An exception class indicating that we could not parse the input text
    """
    pass
