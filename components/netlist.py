from collections import OrderedDict
from PySpice.Tools.StringTools import join_list, join_dict, join_lines
import os

class MyNetlist(OrderedDict):

    """reduce parse time by storing subcircuits as string

    Args:
        OrderedDict (_type_): _description_
        
    """

    def __init__(self, name, is_subcircuit:bool = False, lib = None, options = None):
        super().__init__()
        self._subcircuits = OrderedDict()
        self.is_subcircuit = is_subcircuit
        self.name = name
        self.lib = lib # alsolute library path
        self.footer = ".options TEMP = 25\n.options TNOM = 25\n.op\n.end" \
            if options is None else options
    def __str__(self):
        # if self is main
        netlist = ''
        if not self.is_subcircuit:
            netlist += '.title' + self.name + os.linesep
            netlist += '.include' + self.lib + os.linesep
        # include subcircuits

        netlist = '.subckt ' + join_list((self._name, nodes, parameters)) + os.linesep
        netlist += super().__str__()
        netlist += '.ends ' + self.name + os.linesep

        return netlist
    """
    def add_subcircuit(self, subcircuit):
        assert subcircuit is MyNetlist, 'subcircuit must be Netlist instance'
        self._subcircuits[str(subcircuit.name)] = subcircuit

    def remove_subcircuit(self, name):
        res = self._subcircuits.get(str(name))
        assert res is not None, "invalid subcircuit name"
        self._subcircuits.pop(str(name))
    """
