import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclasses.dataclass
class ParseState:
    """
    Tracks the current state of the parser for error reporting.

    This class is used to track the current state of the parser for error
    reporting.

    You should not need to use this class directly in most cases unless
    building an error handler. Using the `GrammarErrorHandler` class will
    automatically orchestrate the creation of `ParseState`s and their use
    in error reporting.
    """
    current_state: int = 0
    token_stack: List[Any] = dataclasses.field(default_factory=list)
    state_stack: List[int] = dataclasses.field(default_factory=list)
    lookahead: Optional[Any] = None
    valid_shifts: Set[int] = dataclasses.field(default_factory=set)
    valid_reduces: Set[Tuple[int, int]] = dataclasses.field(
        default_factory=set
    )
    error_context: List[Any] = dataclasses.field(
        default_factory=list
    )

    def update(self, parser: Any) -> None:
        """Update the state with current parser information."""
        self.current_state = parser.state
        self.token_stack = parser.symstack[1:]  # Skip initial state
        self.state_stack = parser.statestack
        # PLY doesn't always have lookahead available
        self.lookahead = getattr(parser, 'lookahead', None)

        # Update valid actions
        actions = parser.action[self.current_state]
        self.valid_shifts = {
            s
            for t, s in actions.items()
            if t != 'error' and s > 0
        }
        self.valid_reduces = {
            (-r, self.current_state)
            for t, r in actions.items()
            if t != 'error' and r < 0
        }

        # Keep last few tokens for context
        self.error_context = self.token_stack[-3:] if self.token_stack else []


@dataclasses.dataclass
class ErrorAnalyzer:
    """
    Analyzes parser state to provide error information and recovery
    suggestions.

    This class is used to analyze the current state of the parser to provide
    error information and recovery suggestions.

    You should not need to use this class directly in most cases unless
    building an error handler. Using the `GrammarErrorHandler` class will
    automatically orchestrate the creation of `ErrorAnalyzer`s and their
    use in error reporting.
    """
    grammar: Any
    state_machine: Any

    def analyze_error(self, parse_state: ParseState) -> Dict[str, Any]:
        """Analyze the current parser state to provide error information."""
        # Get valid tokens at current state from action table
        valid_tokens = set()
        token_examples = {}  # Map token names to example values
        for t, s in self.state_machine.action[
            parse_state.current_state
        ].items():
            if t != 'error' and s > 0:
                valid_tokens.add(t)
                # Generate example for this token
                # if we can find it in the grammar
                if hasattr(self.grammar, 'tokens'):
                    for token_def in self.grammar.tokens:
                        if hasattr(token_def, 'name') and token_def.name == t:
                            if hasattr(token_def, 'generate_example'):
                                token_examples[t] = (
                                    token_def.generate_example()
                                )
                            break

        # Get valid production completions by analyzing state stack
        valid_completions = []
        current_state = parse_state.current_state

        # Get all productions that could be reduced in current state
        for t, r in self.state_machine.action[current_state].items():
            if r < 0:  # Negative numbers indicate reductions
                prod = self.state_machine.grammar.productions[-r]
                rhs = prod.rule.split()[2:]  # Skip LHS and ':'

                # Check if current stack matches start of this production
                if len(parse_state.token_stack) > 0:
                    matched = True
                    for i, token in enumerate(parse_state.token_stack):
                        if i >= len(rhs) or token.type != rhs[i]:
                            matched = False
                            break

                    if matched:
                        # Format completion example with concrete values
                        matched_tokens = [
                            t.value for t in parse_state.token_stack
                        ]
                        remaining = rhs[len(matched_tokens):]
                        if remaining:
                            # Generate concrete examples for remaining tokens
                            concrete_remaining = []
                            for token_name in remaining:
                                if token_name in token_examples:
                                    concrete_remaining.append(
                                        token_examples[token_name]
                                    )
                                else:
                                    concrete_remaining.append(token_name)

                            valid_completions.append({
                                'production': prod.rule,
                                'matched': matched_tokens,
                                'remaining': remaining,
                                'concrete_remaining': concrete_remaining,
                            })

        # Get context from state stack
        context = []
        for state in parse_state.state_stack:
            # Get all possible reductions in this state
            for t, r in self.state_machine.action[state].items():
                if r < 0:
                    prod = self.state_machine.grammar.productions[-r]
                    context.append(prod.name)

        return {
            'valid_tokens': valid_tokens,
            'valid_completions': valid_completions,
            'context': context,
            'token_examples': token_examples,
        }


@dataclasses.dataclass
class RecoveryStrategy:
    """Handles error recovery strategies for the parser.

    This class is used to attempt to recover from an error using various
    strategies.

    You should not need to use this class directly in most cases unless
    building an error handler. Using the `GrammarErrorHandler` class will
    automatically orchestrate the creation of `RecoveryStrategy`s and their
    use in error reporting.
    """
    grammar: Any
    error_analyzer: ErrorAnalyzer

    def attempt_recovery(
        self,
        parse_state: ParseState,
    ) -> Optional[Dict[str, Any]]:
        """Attempt to recover from an error using various strategies."""
        # Try panic mode recovery first
        if result := self._panic_mode_recovery(parse_state):
            return result

        # Try phrase level recovery
        if result := self._phrase_level_recovery(parse_state):
            return result

        # Try error production recovery
        if result := self._error_production_recovery(parse_state):
            return result

        return None

    def _panic_mode_recovery(
        self,
        parse_state: ParseState,
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt panic mode recovery by skipping tokens until a synchronization
        token is found.
        """
        # Get all states that can be reached from current state
        reachable_states = set()
        # Use the error analyzer's state machine instead of grammar
        for t, s in self.error_analyzer.state_machine.action[
            parse_state.current_state
        ].items():
            if s > 0:  # Positive numbers indicate shifts
                reachable_states.add(s)

        # Find synchronization tokens by looking at what tokens are accepted
        # in any reachable state
        sync_tokens = set()
        for state in reachable_states:
            for t, s in (
                self.error_analyzer.state_machine.action[state].items()
            ):
                if t != 'error' and s > 0:
                    sync_tokens.add(t)

        if sync_tokens:
            return {
                'strategy': 'panic_mode',
                'sync_tokens': sync_tokens,
                'action': 'skip_until_sync',
            }
        return None

    def _phrase_level_recovery(
        self,
        parse_state: ParseState,
    ) -> Optional[Dict[str, Any]]:
        """Attempt phrase level recovery by trying to fix common mistakes."""
        raise NotImplementedError("Phrase level recovery not implemented")
        # # Get valid tokens and completions
        # analysis = self.error_analyzer.analyze_error(parse_state)

        # # Check for common mistakes by looking at state transitions
        # common_mistakes = {
        #     'missing_paren': ('LPAREN', 'RPAREN'),
        #     'missing_bracket': ('LBRACKET', 'RBRACKET'),
        #     'missing_brace': ('LBRACE', 'RBRACE'),
        #     'missing_comma': ('COMMA',),
        #     'missing_semicolon': ('SEMICOLON',),
        # }

        # for mistake_type, tokens in common_mistakes.items():
        #     if all(t in analysis['valid_tokens'] for t in tokens):
        #         return {
        #             'strategy': 'phrase_level',
        #             'mistake_type': mistake_type,
        #             'action': 'insert_tokens',
        #             'tokens': tokens,
        #         }

        # return None

    def _error_production_recovery(
        self,
        parse_state: ParseState,
    ) -> Optional[Dict[str, Any]]:
        """Attempt recovery by inserting error tokens and continuing."""
        # Get valid completions
        analysis = self.error_analyzer.analyze_error(parse_state)

        if analysis['valid_completions']:
            # Find the most likely completion based on state transitions
            best_completion = min(
                analysis['valid_completions'],
                key=lambda c: len(c['remaining']),
            )

            return {
                'strategy': 'error_production',
                'action': 'insert_error_token',
                'completion': best_completion,
            }

        return None


@dataclasses.dataclass(frozen=True)
class GrammarErrorHandler:
    """
    Handler for grammar-level errors.

    Attributes
    ----------
    token_error: Optional[callable]
        Function to handle lexer errors.
    parser_error: Optional[callable]
        Function to handle parser errors.
    error_contexts: Dict[str, str]
        Mapping of error contexts to messages.
    example_values: Dict[str, str]
        Mapping of token types to example values.
    _parser: Optional[Any]
        Reference to the parser instance.
    """
    token_error: Optional[callable] = None
    parser_error: Optional[callable] = None
    error_contexts: Dict[str, str] = dataclasses.field(
        default_factory=dict
    )
    example_values: Dict[str, str] = dataclasses.field(
        default_factory=dict
    )
    _parser: Optional[Any] = None
    _parse_state: Optional[ParseState] = None
    _error_analyzer: Optional[ErrorAnalyzer] = None
    _recovery_strategy: Optional[RecoveryStrategy] = None

    def __post_init__(self):
        """Validate the error handler."""
        if self.token_error and not callable(self.token_error):
            raise ValueError("Token error handler must be callable")
        if self.parser_error and not callable(self.parser_error):
            raise ValueError("Parser error handler must be callable")
        if not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in self.error_contexts.items()
        ):
            raise ValueError("Error contexts must be string-string pairs")
        if not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in self.example_values.items()
        ):
            raise ValueError("Example values must be string-string pairs")

    def materialise_examples(
        self,
        tokens: Tuple[Any, ...],
        precomputed: Dict[str, str] = None,
    ) -> 'GrammarErrorHandler':
        """Generate and cache example values for tokens."""
        if precomputed is None:
            precomputed = {}
        example_values = {
            token.name: precomputed.get(token.name, token.generate_example())
            for token in tokens
        }
        return dataclasses.replace(
            self,
            example_values=example_values
        )

    def _set_parser(self, parser: Any) -> None:
        """
        Set the parser reference and initialize error handling components.
        """
        object.__setattr__(self, '_parser', parser)
        object.__setattr__(self, '_parse_state', ParseState())
        object.__setattr__(
            self,
            '_error_analyzer',
            ErrorAnalyzer(parser.grammar, parser)
        )
        object.__setattr__(
            self,
            '_recovery_strategy',
            RecoveryStrategy(parser.grammar, self._error_analyzer)
        )

        # Build token type to category mapping
        token_categories = {}
        if hasattr(parser.grammar, 'components'):
            # Get the original token definitions from the grammar components
            for component in parser.grammar.components:
                for token_def in component.tokens:
                    if (
                        hasattr(token_def, 'name') and
                        hasattr(token_def, 'category')
                    ):
                        token_categories[token_def.name] = token_def.category
        object.__setattr__(self, '_token_categories', token_categories)

    def create_token_error_function(self) -> callable:
        """Create a PLY-compatible token error function."""
        if self.token_error:
            return self.token_error
        # Default token error handler with context
        def t_error(t):
            # Get the line and column information
            try:
                lexdata = t.lexer.lexdata
                pos = t.lexpos
                # Compute line number and line start
                line = lexdata.count('\n', 0, pos) + 1
                line_start = lexdata.rfind('\n', 0, pos) + 1
                line_end = lexdata.find('\n', pos)
                if line_end == -1:
                    line_end = len(lexdata)
                line_content = lexdata[line_start:line_end]
                value = t.value
                # If the error token is more than one character, point to the
                # first character
                if isinstance(value, str) and len(value) > 1:
                    col = pos - line_start + 1
                else:
                    col = pos - line_start + 1
                # For multiline, skip leading whitespace/newlines in
                # line_content
                if not line_content.strip():
                    # Find the next non-empty line
                    lines = lexdata.splitlines()
                    for i, l in enumerate(lines, 1):
                        if l.strip() and i >= line:
                            line_content = l
                            line = i
                            line_start = lexdata.find(l)
                            col = pos - line_start + 1
                            break
                pointer = ' ' * (col - 1) + '^'
                context = next(
                    (
                        msg for ctx, msg in self.error_contexts.items()
                        if ctx in t.type
                    ),
                    f"Illegal character '{value[0]}'"
                )
                error_msg = (
                    f"Lexical error at line {line}, column {col}:\n"
                    f"{line_content}\n"
                    f"{pointer}\n"
                    f"{context}"
                )
            except Exception:
                # Fallback if we can't get line context
                value = getattr(t, 'value', '?')
                pos = getattr(t, 'lexpos', '?')
                line = getattr(t, 'lineno', '?')
                error_msg = (
                    f"Lexical error at line {line}, position {pos}: "
                    f"Illegal character '{value[0] if value else '?'}'"
                )
            raise ValueError(error_msg)
        return t_error

    def create_parser_error_function(self) -> callable:
        """Create a PLY-compatible parser error function."""
        if self.parser_error:
            return self.parser_error

        # Default parser error handler with enhanced error reporting
        def p_error(p):
            if p is None:  # EOF case
                if self._parser is None:
                    raise ValueError(
                        "Error handler is missing a parser reference"
                    )

                # Update parse state
                self._parse_state.update(self._parser)
                analysis = self._error_analyzer.analyze_error(
                    self._parse_state,
                )
                error_msg = ["Unexpected end of input"]

                if self._parse_state.token_stack:
                    last_token = self._parse_state.token_stack[-1]
                    # Try to get line, column, and line content
                    try:
                        lexdata = last_token.lexer.lexdata
                        pos = last_token.lexpos
                        line = lexdata.count('\n', 0, pos) + 1
                        line_start = lexdata.rfind('\n', 0, pos) + 1
                        line_end = lexdata.find('\n', pos)
                        if line_end == -1:
                            line_end = len(lexdata)
                        line_content = lexdata[line_start:line_end]
                        col = pos - line_start + len(str(last_token.value))
                        pointer = ' ' * (col - 1) + '^'
                        error_msg.append(f"at line {line}, column {col}:")
                        error_msg.append(f"{line_content}")
                        error_msg.append(f"{pointer}")
                        error_msg.append(
                            f"Last valid token was '{last_token.value}' of "
                            f"type '{last_token.type}'"
                        )
                        # Add context-specific message if available
                        context = self._get_error_context(
                            last_token,
                            self._parse_state,
                        )
                        if context:
                            error_msg.append(f"\nContext: {context}")
                    except Exception:
                        error_msg.append(
                            f"Last valid token was '{last_token.value}' of "
                            f"type '{last_token.type}'"
                        )
                else:
                    error_msg.append("No valid tokens were parsed.")

                # Add expected tokens (always show section for consistency)
                error_msg.append("\nExpected one of:")
                if analysis['valid_tokens']:
                    for token_name in analysis['valid_tokens']:
                        if token_name in analysis.get('token_examples', {}):
                            example = analysis['token_examples'][token_name]
                            error_msg.append(
                                f"  - {token_name} (e.g., '{example}')"
                            )
                        else:
                            error_msg.append(f"  - {token_name}")
                else:
                    error_msg.append("  (no valid tokens)")

                # Add valid completions (always show section for consistency)
                error_msg.append("\nValid completions could be:")
                if analysis['valid_completions']:
                    for completion in analysis['valid_completions'][:3]:
                        matched = ' '.join(completion['matched'])
                        if 'concrete_remaining' in completion:
                            remaining = ' '.join(
                                completion['concrete_remaining']
                            )
                        else:
                            remaining = ' '.join(completion['remaining'])
                        error_msg.append(f"  {matched} {remaining}")
                else:
                    error_msg.append("  (no valid completions)")

                # Add recovery suggestion if available
                recovery = self._recovery_strategy.attempt_recovery(
                    self._parse_state
                )
                if recovery:
                    if recovery['strategy'] == 'panic_mode':
                        error_msg.append(
                            f"\nRecovery: Skip until one of: "
                            f"{', '.join(recovery['sync_tokens'])}"
                        )
                    elif recovery['strategy'] == 'phrase_level':
                        error_msg.append(
                            f"\nRecovery: Insert missing "
                            f"{recovery['mistake_type']}"
                        )
                    elif recovery['strategy'] == 'error_production':
                        error_msg.append(
                            f"\nRecovery: Complete as: "
                            f"{recovery['completion']['production']}"
                        )

                # Add context-specific message if available
                context = self._get_error_context(
                    last_token,
                    self._parse_state,
                )
                if context:
                    error_msg.append(f"\nContext: {context}")

                error_msg = "\n".join(error_msg)
                raise ValueError(error_msg)
            else:
                # Non-EOF case
                # Get token information
                token = p
                line = token.lineno
                pos = token.lexpos
                value = token.value
                type_ = token.type

                # Update parse state
                self._parse_state.update(self._parser)

                # Try to get the line content for context
                try:
                    # Get the lexer's input
                    lexdata = token.lexer.lexdata
                    # Find the start of the current line
                    line_start = lexdata.rfind('\n', 0, pos) + 1
                    # Find the end of the current line
                    line_end = lexdata.find('\n', pos)
                    if line_end == -1:
                        line_end = len(lexdata)
                    # Get the full line
                    line_content = lexdata[line_start:line_end]
                    # Calculate the column position
                    col = pos - line_start + 1
                    # Create a pointer to the error position
                    pointer = ' ' * (col - 1) + '^'

                    # Analyze error
                    analysis = self._error_analyzer.analyze_error(
                        self._parse_state
                    )

                    # Try recovery
                    recovery = self._recovery_strategy.attempt_recovery(
                        self._parse_state,
                    )

                    # Format the error message
                    error_msg = [
                        f"Syntax error at line {line}, column {col}:",
                        f"{line_content}",
                        f"{pointer}",
                        f"Unexpected token '{value}' of type '{type_}'"
                    ]

                    # Add expected tokens (always show section for
                    # consistency)
                    error_msg.append("\nExpected one of:")
                    if analysis['valid_tokens']:
                        for token_name in analysis['valid_tokens']:
                            if token_name in analysis.get(
                                'token_examples', {}
                            ):
                                example = analysis[
                                    'token_examples'
                                ][token_name]
                                error_msg.append(
                                    f"  - {token_name} (e.g., '{example}')"
                                )
                            else:
                                error_msg.append(f"  - {token_name}")
                    else:
                        error_msg.append("  (no valid tokens)")

                    # Add valid completions (always show section for
                    # consistency)
                    error_msg.append("\nValid completions could be:")
                    if analysis['valid_completions']:
                        for completion in analysis['valid_completions'][:3]:
                            matched = ' '.join(completion['matched'])
                            if 'concrete_remaining' in completion:
                                remaining = ' '.join(
                                    completion['concrete_remaining']
                                )
                            else:
                                remaining = ' '.join(completion['remaining'])
                            error_msg.append(f"  {matched} {remaining}")
                    else:
                        error_msg.append("  (no valid completions)")

                    # Add recovery suggestion if available
                    if recovery:
                        if recovery['strategy'] == 'panic_mode':
                            error_msg.append(
                                f"\nRecovery: Skip until one of: "
                                f"{', '.join(recovery['sync_tokens'])}"
                            )
                        elif recovery['strategy'] == 'phrase_level':
                            error_msg.append(
                                f"\nRecovery: Insert missing "
                                f"{recovery['mistake_type']}"
                            )
                        elif recovery['strategy'] == 'error_production':
                            error_msg.append(
                                f"\nRecovery: Complete as: "
                                f"{recovery['completion']['production']}"
                            )

                    # Add context-specific message if available
                    context = self._get_error_context(
                        token,
                        self._parse_state,
                    )
                    if context:
                        error_msg.append(f"\nContext: {context}")

                    error_msg = "\n".join(error_msg)
                    raise ValueError(error_msg)
                except (AttributeError, IndexError, Exception):
                    # Fallback if we can't get line context
                    # Try to get line and column and line content if possible
                    line = getattr(token, 'lineno', '?')
                    pos = getattr(token, 'lexpos', '?')
                    col = None
                    line_content = None
                    pointer = None
                    if (
                        hasattr(token, 'lexer') and
                        hasattr(token.lexer, 'lexdata') and
                        pos != '?'
                    ):
                        lexdata = token.lexer.lexdata
                        line_start = lexdata.rfind('\n', 0, pos) + 1
                        col = pos - line_start + 1
                        line_end = lexdata.find('\n', pos)
                        if line_end == -1:
                            line_end = len(lexdata)
                        line_content = lexdata[line_start:line_end]
                        pointer = ' ' * (col - 1) + '^'

                    # Analyze error for fallback
                    analysis = self._error_analyzer.analyze_error(
                        self._parse_state,
                    )

                    if (
                        col is not None and
                        line_content is not None and
                        pointer is not None
                    ):
                        error_msg = [
                            f"Syntax error at line {line}, column {col}:",
                            f"{line_content}",
                            f"{pointer}",
                            f"Unexpected token '{value}' of type '{type_}'"
                        ]
                    elif col is not None:
                        error_msg = [
                            f"Syntax error at line {line}, column {col}: "
                            f"Unexpected token '{value}' of type '{type_}'"
                        ]
                    else:
                        error_msg = [
                            f"Syntax error at line {line}, position {pos}: "
                            f"Unexpected token '{value}' of type '{type_}'"
                        ]

                    # Add expected tokens (always show section for
                    # consistency)
                    error_msg.append("\nExpected one of:")
                    if analysis['valid_tokens']:
                        for token_name in analysis['valid_tokens']:
                            if token_name in analysis.get(
                                'token_examples',
                                {},
                            ):
                                example = analysis[
                                    'token_examples'
                                ][token_name]
                                error_msg.append(
                                    f"  - {token_name} (e.g., '{example}')"
                                )
                            else:
                                error_msg.append(f"  - {token_name}")
                    else:
                        error_msg.append("  (no valid tokens)")

                    # Add valid completions (always show section for
                    # consistency)
                    error_msg.append("\nValid completions could be:")
                    if analysis['valid_completions']:
                        for completion in analysis['valid_completions'][:3]:
                            matched = ' '.join(completion['matched'])
                            if 'concrete_remaining' in completion:
                                remaining = ' '.join(
                                    completion['concrete_remaining']
                                )
                            else:
                                remaining = ' '.join(completion['remaining'])
                            error_msg.append(f"  {matched} {remaining}")
                    else:
                        error_msg.append("  (no valid completions)")

                    # Add context-specific message if available
                    context = self._get_error_context(
                        token,
                        self._parse_state,
                    )
                    if context:
                        error_msg.append(f"\nContext: {context}")

                    error_msg = "\n".join(error_msg)
                    raise ValueError(error_msg)

        return p_error

    def _get_error_context(
        self,
        token: Any,
        parse_state: Optional[ParseState] = None,
    ) -> Optional[str]:
        """
        Get error context using multiple strategies in order of preference.
        """

        # 1. Token category (most specific) - use token type to category
        #    mapping
        if (
            hasattr(self, '_token_categories') and
            token.type in self._token_categories
        ):
            category = self._token_categories[token.type]
            if category and category in self.error_contexts:
                return self.error_contexts[category]

        # 2. Production context (contextual) - if we have parse state
        if parse_state:
            prod_context = self._get_error_context_from_production(
                parse_state,
            )
            if prod_context:
                return prod_context

        # 3. State context (fallback) - if we have parse state
        if parse_state:
            state_context = self._get_error_context_from_state(
                parse_state.current_state,
            )
            if state_context:
                return self.error_contexts.get(state_context)

        # 4. Substring matching (legacy fallback)
        for ctx, msg in self.error_contexts.items():
            if ctx in token.type:
                return msg

        return None

    def _get_error_context_from_production(
        self,
        parse_state: ParseState,
    ) -> Optional[str]:
        """Get context based on what production rule was being built."""
        if not hasattr(self, '_parser') or self._parser is None:
            return None

        # Analyze state stack to determine context
        for state in parse_state.state_stack:
            for t, r in self._parser.action[state].items():
                if r < 0:  # Reduction
                    prod = self._parser.grammar.productions[-r]
                    if prod.name in self.error_contexts:
                        return self.error_contexts[prod.name]
        return None

    def _get_error_context_from_state(
        self,
        current_state: int,
    ) -> Optional[str]:
        """Get context based on parser state."""
        # Map states to contexts - this could be made configurable
        state_contexts = {
            # Add state-to-context mappings as needed
            # Example: 5: "EXPRESSION_CONTEXT",
            # Example: 12: "CONDITION_CONTEXT",
        }
        return state_contexts.get(current_state)
