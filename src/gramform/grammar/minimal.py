import dataclasses
from typing import Any, Tuple

import ply.lex as lex
import ply.yacc as yacc
import wadler_lindig as wl


PRIMITIVES = {}
_RESERVED = {
    'I_': 'INDICATOR',
    'd_': 'BACKDIFF',
    'dd_': 'BACKDIFF_INCLUSIVE',
}


@dataclasses.dataclass(frozen=True)
class Primitive:
    name: str
    parameters: Tuple[Any, ...] = dataclasses.field(
        default_factory=tuple
    )
    # Used for operator flattening when postprocessing the tree.
    is_associative: bool = False

    def __post_init__(self):
        # Wasteful to do this every time we bind.
        PRIMITIVES[self.name] = type(self)

    def bind(self, *pparams):
        pparams = pparams or ()
        return type(self)(name=self.name, parameters=tuple(pparams))

    def __repr__(self):
        return wl.pformat(self)


CONCATENATE = Primitive("CONCATENATE", is_associative=True)
POWER = Primitive("POWER")
BACKDIFF = Primitive("BACKDIFF")
RANGE = Primitive("RANGE")
ENUM = Primitive("ENUM")
INDICATOR = Primitive("INDICATOR")
CONDITION_EQUAL = Primitive("CONDITION_EQUAL", is_associative=True)
CONDITION_NOT_EQUAL = Primitive("CONDITION_NOT_EQUAL", is_associative=True)
CONDITION_LESS = Primitive("CONDITION_LESS")
CONDITION_LESS_EQUAL = Primitive("CONDITION_LESS_EQUAL")
CONDITION_GREATER = Primitive("CONDITION_GREATER")
CONDITION_GREATER_EQUAL = Primitive("CONDITION_GREATER_EQUAL")
VARIABLE = Primitive("VARIABLE")


def confound_formula_preprocessor():
    return {
        'wm': 'white_matter',
        'gsr': 'global_signal',
        'gs': 'global_signal',
        'rps': 'trans_x + trans_y + trans_z + rot_x + rot_y + rot_z',
        'fd': 'framewise_displacement',
        'dv': 'std_dvars',
        'acc': 'a_comp_cor',
        'wcc': 'w_comp_cor',
        'ccc': 'c_comp_cor',
    }


class MinimalGrammar:
    tokens = (
        'CONCATENATE',
        'POWER',
        'POWER_INCLUSIVE',
        'RANGE',
        'BACKDIFF',
        'BACKDIFF_INCLUSIVE',
        'ENUM',
        'INDICATOR',
        'CONDITION_EQUAL',
        'CONDITION_NOT_EQUAL',
        'CONDITION_LESS',
        'CONDITION_LESS_EQUAL',
        'CONDITION_GREATER',
        'CONDITION_GREATER_EQUAL',
        'LPAREN',
        'RPAREN',
        'LBRACKET',
        'RBRACKET',
        'LBRACE',
        'RBRACE',
        'VARIABLE',
        'FLOAT',
        'INTEGER',
    )

    t_CONCATENATE = r'\+'
    t_POWER = r'\^'
    t_POWER_INCLUSIVE = r'\^\^'
    t_RANGE = r'\-'
    t_ENUM = r','
    t_CONDITION_EQUAL = r'='
    t_CONDITION_NOT_EQUAL = r'(<>|!=|~=)'
    t_CONDITION_LESS = r'<'
    t_CONDITION_LESS_EQUAL = r'<='
    t_CONDITION_GREATER = r'>'
    t_CONDITION_GREATER_EQUAL = r'>='
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_FLOAT = r'\d+\.\d*'
    t_INTEGER = r'\d+'
    t_ignore = ' \t'

    def t_VARIABLE(t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'
        if t.value in _RESERVED:
            t.type = _RESERVED[t.value]
        return t

    def t_error(t):
        raise ValueError(f"Illegal character '{t.value[0]}'")
    
    precedence = (
        (
            'left',
            'CONDITION_EQUAL',
            'CONDITION_NOT_EQUAL',
            'CONDITION_LESS',
            'CONDITION_LESS_EQUAL',
            'CONDITION_GREATER',
            'CONDITION_GREATER_EQUAL',
        ),
        ('left', 'CONCATENATE'),
        ('left', 'POWER', 'POWER_INCLUSIVE'),
        ('left', 'RANGE'),
    )

    def p_expression_concatenate(p):
        'expression : expression CONCATENATE expression'
        p[0] = CONCATENATE.bind(p[1], p[3])

    def p_expression_power(p):
        'expression : expression POWER expression'
        p[0] = POWER.bind(p[1], p[3])

    def p_expression_power_inclusive(p):
        'expression : expression POWER_INCLUSIVE expression'
        p[0] = POWER.bind(p[1], RANGE.bind(1, p[3]))

    def p_expression_backdiff(p):
        'expression : BACKDIFF parameter LPAREN expression RPAREN'
        p[0] = BACKDIFF.bind(p[4], p[2])

    def p_expression_backdiff_inclusive(p):
        'expression : BACKDIFF_INCLUSIVE parameter LPAREN expression RPAREN'
        p[0] = BACKDIFF.bind(p[4], RANGE.bind(0, p[2]))

    def p_expression_range(p):
        'expression : expression RANGE expression'
        p[0] = RANGE.bind(p[1], p[3])

    def p_expression_indicator(p):
        'expression : INDICATOR parameter'
        p[0] = INDICATOR.bind(p[2])

    def p_expression_condition_equal(p):
        'expression : expression CONDITION_EQUAL expression'
        p[0] = CONDITION_EQUAL.bind(p[1], p[3])

    def p_expression_condition_not_equal(p):
        'expression : expression CONDITION_NOT_EQUAL expression'
        p[0] = CONDITION_NOT_EQUAL.bind(p[1], p[3])

    def p_expression_condition_less(p):
        'expression : expression CONDITION_LESS expression'
        p[0] = CONDITION_LESS.bind(p[1], p[3])

    def p_expression_condition_less_equal(p):
        'expression : expression CONDITION_LESS_EQUAL expression'
        p[0] = CONDITION_LESS_EQUAL.bind(p[1], p[3])

    def p_expression_condition_greater(p):
        'expression : expression CONDITION_GREATER expression'
        p[0] = CONDITION_GREATER.bind(p[1], p[3])

    def p_expression_condition_greater_equal(p):
        'expression : expression CONDITION_GREATER_EQUAL expression'
        p[0] = CONDITION_GREATER_EQUAL.bind(p[1], p[3])

    def p_expression_paren_term(p):
        'expression : LPAREN expression RPAREN'
        p[0] = p[2]

    def p_expression_parameter(p):
        'parameter : LBRACKET expression RBRACKET'
        p[0] = p[2]

    def p_expression_term_variable(p):
        'expression : VARIABLE'
        p[0] = VARIABLE.bind(p[1])

    def p_expression_term_integer(p):
        'expression : INTEGER'
        p[0] = int(p[1])

    def p_expression_term_float(p):
        'expression : FLOAT'
        p[0] = float(p[1])

    def p_error(p):
        raise ValueError(f"Syntax error: {p}")


def MinimalGrammarLexer(**params):
    lexer = lex.lex(module=MinimalGrammar, **params)
    return lexer


def MinimalGrammarParser(**params):
    parser = yacc.yacc(module=MinimalGrammar, **params)
    return parser


def main():
    #expr = '(x+y+z)^^2+(x+y+z)+((x+y+z)^2+(x+y+z))^3.13-5'
    expr = '(x+y+z)^^2-3 + I_[x=y] + dd_[2](x)'
    lexer = MinimalGrammarLexer()
    parser = MinimalGrammarParser()
    lexer.input(expr)
    for tok in lexer:
        print(tok)
    result = parser.parse(expr)
    print(result)
    breakpoint()


if __name__ == "__main__":
    main()
