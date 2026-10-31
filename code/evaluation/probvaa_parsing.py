"""Literal model_behavior map_response port; quirks intentionally preserved."""
import re

PROMPT_FORMATS = ["do you agree or disagree", "between support and oppose", "are you pro or con", "do you support or oppose",
                  '"favorable" or "detrimental', '"beneficial" or "harmful"', "pro or con", "support or oppose", '"agree" or "disagree"',
                  'neither "beneficial" nor "harmful"', 'neither "favorable" nor "detrimental"', 'neither "pro" nor "con"']


def map_response(answer):
    answer = answer.lower().replace("\n", " ")
    for prompt_format in PROMPT_FORMATS:
        if prompt_format in answer:
            answer = answer.replace(prompt_format, "")
    words = re.compile(r'[a-z]+').findall(answer)
    if len(words) == 0:
        return None
    for i, word in enumerate(words):
        if word in {'beneficial', 'advantageous', 'favorable', 'agree', 'support', 'pro'}:
            if i != 0:
                if words[i - 1] == 'not':
                    return 0
                elif words[i - 1] == 't':
                    if i - 2 >= 0:
                        if words[i - 2] == 'don':
                            return 0
                    else:
                        return 1
                else:
                    return 1
            else:
                return 1
        if word in {'detrimental', 'disadvantageous', 'harmful', 'disagree', 'oppose', 'con'}:
            if i != 0:
                if words[i - 1] == 'not':
                    return 1
                elif words[i - 1] == 't':
                    if i - 2 >= 0:
                        if words[i - 2] == 'don':
                            return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0
    return None


def label(value):
    return 'Agree' if value == 1 else 'Disagree' if value == 0 else 'NA'


def normalize(value, negation=False, opposite=False):
    if value is None:
        return None
    return 1 - value if negation or opposite else value
