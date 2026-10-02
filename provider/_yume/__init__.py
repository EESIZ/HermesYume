"""hermesyume provider helper package (stdlib only, no import-time side effects).

Lives in a sub-package because the Hermes loader pre-executes sibling ``*.py`` files of the
provider directory and silently swallows their failures (PLAN-v2 §6.1).
"""
