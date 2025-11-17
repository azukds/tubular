Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Each section will have a title of the format ``X.Y.Z (YYYY-MM-DD)`` giving the version of the package and the date of release of that version. Unreleased changes i.e. those that have been merged into `main` (e.g. with a .dev suffix) but which are not yet in a new release (on pypi) are added to the changelog but with the title ``X.Y.Z (unreleased)``. Unreleased sections can be combined when they are released and the date of release added to the title.

Prior to version 2.*, tubular versioning practices were not consistent. Going forwards, we would like developers to stick to semantic versioning rules, described below.

Semantic versioning follows a pattern of MAJOR.MINOR.PATCH where each part represents a specific type of change:

- MAJOR: Incremented for incompatible API changes.

- MINOR: Incremented for added functionality in a backward-compatible manner.

- PATCH: Incremented for backward-compatible bug fixes.

This structure allows developers and users to understand the potential impact of updating to a new version at a glance.

We use the tags:
- feat: new or improved functionality
- bug: fix to existing functionality
- chore: minor improvements to repo health

Each individual change should have a link to the pull request after the description of the change.

2.3.0 (unreleased)
------------------

Changed
^^^^^^^

- feat: added `to_json` method for `BaseAggregationTransformer` `#610 <https://github.com/azukds/tubular/issues/610>`_
- feat: added `to_json` method for `AggregateRowsOverColumnTransformer` `#611 <https://github.com/azukds/tubular/issues/611>`_
- feat: converted BaseTransfomer to support lazyframes, and added lazyframe testing `#535 <https://github.com/azukds/tubular/issues/535>_`
- feat: added lazyframe testing for BaseTransfomer
- feat: introduced `lazyframe_compatible` class attr to all transformers
- feat: as part of lazyframe work, transformers no longer error for emptyframes (they just return emptyframes)
- bugfix: MeanResponseTransformer approach was hitting a recursion depth limit error for many levels, have switched to more resilient (and generally better) approach
- chore: add beartype decorator to transformers NullIndicator and SetValueTransformer - #563 <https://github.com/azukds/tubular/issues/563>
- feat: optimisation changes to BaseCappingTransformer `#484 <https://github.com/azukds/tubular/issues/484>`

2.2.0 (11/11/2025)
------------------

Changed
^^^^^^^

- feat: added `to_json` method for `SetValueTransformer` `#542 <https://github.com/azukds/tubular/issues/542>`_
- feat: added 'to_json' method for GroupRareLevelsTransformer '#548 <https://github.com/azukds/tubular/issues/548>'
- removed SeparatorColumnMixin `#562 <https://github.com/azukds/tubular/issues/562>`_
