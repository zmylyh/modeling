import re
def match_pattern(s):
    m = re.match(r"comb_\('(\w+)',_(\d+),_'(\w+)'\)", s)
    return m.group(1), int(m.group(2)), m.group(3)
print(match_pattern("comb_('F2232323234',_29,_'s2')"))