import pprint

pp = pprint.PrettyPrinter(indent=1)

x = {
    "a": 123,
    "b": 22,
    "c": 123,
    "d": 3,
    "e": 77
}
pp.pprint(x)