"""Tools module.

Empty by design.  The DeepAgent orchestrator imports the lightweight
``@tool`` helpers under :mod:`tools.research`, :mod:`tools.navigation`,
and other subpackages directly, so this top-level ``__init__`` keeps no
re-exports to avoid pulling heavy dependencies at package load time.
"""
