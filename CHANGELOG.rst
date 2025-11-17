Changelog
=========

Notes on Versioning Approach
---------------------------

This changelog follows the great advice from https://keepachangelog.com/.

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

- chore: cleanup of ruff rules `#590 <https://github.com/azukds/tubular/issues/590>`_
- feat: added `to_json` method for `SetValueTransformer` `#542 <https://github.com/azukds/tubular/issues/542>`_
- feat: added 'to_json' method for GroupRareLevelsTransformer '#548 <https://github.com/azukds/tubular/issues/548>'
- removed SeparatorColumnMixin `#562 <https://github.com/azukds/tubular/issues/562>_`

2.1.0 (30/10/25)
------------------

Changed
^^^^^^^

- feat: optimisation changes to fit methods in imputers file
- feat: optimisation changes to transform method of NullIndicator
- feat: optimisation changes to MeanResponseTransformer
- feat: created _stats and _checks files to store reusable calculation or data validating functions
- feat: converted DropOriginalMixin and associated transformers to beartype
- feat: removed NewColumnNameMixin and converted associated transformers/tests to beartype
- feat: removed TwoColumnMixin and BaseDateTwoColumnTransformer and replaced with beartype
- feat: added `get_feature_names_out` method to all transformers, which gives back list of modified/created features. This follows the sklearn API.
- feat: added `to_json` method for BaseGenericDateTransformer `#540 <https://github.com/azukds/tubular/issues/540>_`
- feat: added `to_json` method for DateDifferenceTransformer `#552 <https://github.com/azukds/tubular/issues/552>_`
- feat: added `to_json` method for BaseMappingTransformer, MappingTransformer `#541 <https://github.com/azukds/tubular/issues/541>_`

2.0.0 (16/10/2025)
------------------

Changed
^^^^^^^

- chore: refactored unit weight handling into method of WeightColumnMixin, tightened up handling
- feat: added AggregateColumnsOverRowTransformer `#385 <https://github.com/azukds/tubular/issues/385>_`
- chore: renamed AggregateRowOverColumnsTransformer to AggregateRowsOverColumnTransformer
- feat: optimisation changes to GroupRareLevelsTransformer fit method
- feat: optimisation changes to DatetimeSinusoidCalculator, added 'return_native_override' argument to DatetimeSinusoidCalculator, reduced with_columns being called many times. `<#465 <https://github.com/azukds/tubular/issues/465>_`
- chore: turned on doctest
- chore: deprecated DataFrameMethodTransformer
- chore: added doctest examples for BaseTransfomer
- chore: deleted stale example notebooks for BaseTransfomer (replaced by doctest) and DataFrameMethodTransformer (deprecated)
- bugfix: updated minimum narwhals version to 1.42.1 in toml, to avoid import issues for IntoDtype
- chore: deprecated transformers that are not being converted to narwhals, and moved to bottom of their files. `#433 <https://github.com/azukds/tubular/issues/433>_`
- chore: edited package init to only advertise non-deprecated transformers (and not base classes)
- chore: added doctests for capping file and deleted stale example notebooks `#501 <https://github.com/azukds/tubular/issues/501>_`
- chore: added doctests for aggregations file `#500 <https://github.com/azukds/tubular/issues/500>_`
- chore: doctests for misc module, deleted old/stale example notebooks `#505  <https://github.com/azukds/tubular/issues/505>_`
- chore: doctests for mapping module, deleted old/stale example notebooks `#504 <https://github.com/azukds/tubular/issues/504>_`
- chore: doctests for numeric module `#507 <https://github.com/azukds/tubular/issues/507>_`
- feat: added to/from json functionality for BaseTransfomer and imputers file. 
- feat: introduced `jsonable` class attribute across package to control whether json tests run for given class
- feat: introduced _version class attribute that stores package version
- feat: introduced block_from_json decorator to block non-transform related methods for transformers which have been rebuilt from json
- chore: doctests for dates module `#502 <https://github.com/azukds/tubular/issues/502>_`
- chore: doctests for nominal module `#506 <https://github.com/azukds/tubular/issues/506>_`
- chore: doctests for imputers module `#503 <https://github.com/azukds/tubular/issues/503>_`

1.4.8 (03/09/25)
----------------

Changed
^^^^^^^

- feat: optimisation changes to OneHotEncodingTransformer
- feat: optimisation changes to DatetimeSinusoidCalculator, added 'return_native_override' argument to DatetimeSinusoidCalculator, reduced with_columns being called many times. `<#465 https://github.com/azukds/tubular/issues/465>_`
- feat: optimisation changes to AggregateRowsOverColumnTransformer, BaseAggregationTransformer

1.4.7 (21/08/25)
----------------

Changed
^^^^^^^

- feat: narwhalified DatetimeInfoExtractor `#378 <https://github.com/azukds/tubular/issues/378>_`
- feat: optimisations for MeanResponseTransformer, further optimisations for  MappingTransformer `#451 <https://github.com/azukds/tubular/issues/451>_`

1.4.6 (19/08/2025)
------------------

Changed
^^^^^^^

- feat: optimisations for MappingTransformer and BaseMappingTransformerMixin

1.4.5 (19/08/2025)
------------------

Changed
^^^^^^^
- bugfix: updated env to make package importable, added basic test for this
- feat: added BaseAggregationTransformer and AggregateRowOverColumnsTransformer classes in new aggregations module
- narwhalified DatetimeSinusoidCalculator '#425 <https://github.com/azukds/tubular/issues/425>_' 
- Added deprecated warning for DateDiffLeapYearTransformer `#244 <https://github.com/azukds/tubular/issues/244>_`
- Added new units 'week', 'fornight', 'lunar_month', 'common_year' and 'custom_days' to DateDifferenceTransformer. The time component will be truncated for these units and for unit 'D'.
- feat: optimisation changes to BaseTransfomer and imputers file. Edited to reduce number of copies and type changes from to/from_native calls, and select/with_columns being called many times. `#444 <https://github.com/azukds/tubular/issues/444>_`
- feat: added 'return_native' argument to BaseTransfomer to control whether native or narwhals types are returned, and limit type changes. Idea is for this to be rolled out across transformers.
- feat: made creation of copies in BaseTransfomer optional, and default to False.
- feat: optimisations to BaseDatetimeTransformer, BaseDateTransformer, DateDifferenceTransformer, DropOriginalMixin, CheckNumericMixin
- feat: optimisation changes to BaseNominalTransformer, reduced select being called many times, added 'return_native_override' argument. `#450 <https://github.com/azukds/tubular/issues/450>_``
- chore: import narwhals.typing.DType for Github in order to uncap narwhals `#455 <https://github.com/azukds/tubular/issues/455>`
- feat: optimisation changes for GroupRareLevelsTransformer `#446 <https://github.com/azukds/tubular/issues/446>_`
- feat: optimisations to BaseDatetimeTransformer, BaseDateTransformer, DateDifferenceTransformer, DropOriginalMixin
- feat: optimisation changes to BaseNominalTransformer, reduced select being called many times, added 'return_native_override' argument.
- feat: optimisation changes to WeightColumnMixin, combined all weight checks into a single .select call and used narhwals is_nan
- feat: optimisation chnages to BaseCappingTransformer, added 'return_native_override' argument to BaseCappingTransformer and BaseNumericTransformer.
- bugfix: make datetime transformers perform checks on only relevant columns

1.4.4 (24/06/2025)
------------------

Changed
^^^^^^^
- narwhalified ToDatetimeTransformer. Also made some usability improvements, e.g. to accept multiple columns `#379 <https://github.com/azukds/tubular/issues/379>_`
- fixed bug with MappingTransformer, BaseMappingTransformerMixin where nullable boolean mappings were being converted to non-nullable booleans
- Working on above, found additional bug with mapping null values. 
Considered removing this functionality, but it is actually needed for 
inverse pipelines. Changed this part of logic to work more like an imputer.

1.4.3 (02/06/2025)
------------------

Changed
^^^^^^^
- narwhalified ArbitraryImputer `#315 <https://github.com/azukds/tubular/issues/315>_`
- narwhalified BetweenDatesTransformer `#377 <https://github.com/azukds/tubular/issues/377>_`
- feat: narwhalified MeanResponseTransformer `373 <https://github.com/azukds/tubular/issues/373>_`
- narhwalify SetValueTransformer `#398 <https://github.com/azukds/tubular/issues/398>_`
- narwhalified DateDifferenceTransformer. `#376 <https://github.com/azukds/tubular/issues/376>_`
- narwhalified DateDiffLeapYearTransformer.
- narwhalified MappingTransformer `#374 <https://github.com/azukds/tubular/issues/374>_`
- added OneDKmeansTransformer. `#406 <https://github.com/azukds/tubular/issues/406>_`
- beartype typechecking for BaseTransformer init method `#417 <https://github.com/azukds/tubular/issues/417>_`
- narwhalified ToDatetimeTransformer. Also made some usability improvements, e.g. to accept multiple columns `#379 <https://github.com/azukds/tubular/issues/379>_`

1.4.2 (18/03/2025)
------------------

Changed
^^^^^^^

- converted OneHotEncodingTransformer to narwhals `#355 <https://github.com/azukds/tubular/issues/355>_`
- updated WeightsColumnMixin to use new narwhals 'is_finite' method
- narwhalified ModeImputer `#321 <https://github.com/azukds/tubular/issues/321>_`
- fixed issues with all null and nullable-bool column handling in dataframe_init_dispatch
- added NaN error handling to WeightColumnMixin
- narwhalified BaseNumericTransformer `#358 <https://github.com/azukds/tubular/issues/358>_`
- narwhalified BaseCappingTransformer `#357 <https://github.com/azukds/tubular/issues/357>_`
- narwhalified CappingTransformer `#361 <https://github.com/azukds/tubular/issues/361>_`
- narwhalified OutOfRangeNullTransformer `#362 <https://github.com/azukds/tubular/issues/362>_`
- narwhalified MeanImputer `#344 https://github.com/azukds/tubular/issues/344_`
- narwhalified BaseGenericDateTransformer. As part of this updated test data handling of date columns
  across repo `#365 <https://github.com/azukds/tubular/issues/365>_`
- narwhalified BaseNumericTransformer `#358 https://github.com/azukds/tubular/issues/358`
- narwhalified DropOriginalMixin `#352 <https://github.com/azukds/tubular/issues/352>_`
- narwhalified BaseMappingTransformer `#367 <https://github.com/azukds/tubular/issues/367>_`
- narwhalified BaseMappingTransformerMixin. As part of this made mapping transformers more type-conscious, they now rely on an input 'return_dtypes' dict arg.`#369 <https://github.com/azukds/tubular/issues/369>_`
- As part of #369, updated OrdinalEncoderTransformer to output Int8 type
- As part of #369, updated NominalToIntegerTransformer to output Int8 type. Removed inverse_mapping functionality, as this is more complicated when transform is opinionated on types.
- narwhalified GroupRareLevelsTransformer. As part of this, had to make transformer more opinionated and refuse columns with nulls (raises an error directing to imputers.) `#372 <https://github.com/azukds/tubular/issues/372>_`
- narwhalified BaseDatetimeTransformer `#375 <https://github.com/azukds/tubular/issues/375>`
- Optional wanted_levels feature has been integrated into the OneHotEncodingTransformer which allows users to specify which levels in a column they wish to encode. `#384 <https://github.com/azukds/tubular/issues/384>_`
- Created unit tests to check if the values provided for wanted_values are as expected and if the output is as expected.
- fix: issue with falsey values not imputing for ArbitraryImputer `#391 <https://github.com/azukds/tubular/issues/391>_`

1.4.1 (02/12/2024)
------------------

Changed
^^^^^^^

- Refactored BaseImputer to utilise narwhals `#314 <https://github.com/azukds/tubular/issues/314>_`
- Converted test dfs to flexible pandas/polars setup
- Converted BaseNominalTransformer to utilise narwhals `#334 <https://github.com/azukds/tubular/issues/334>_`
- narwhalified CheckNumericMixin `#336 <https://github.com/azukds/tubular/issues/336>_`
- Changed behaviour of NearestMeanResponseImputer so that if there are no nulls at fit, 
  it warns and has no effect at transform, as opposed to erroring. The error was problematic for e.g.
  lightweight test runs where nulls are less likely to be present.

1.4.0 (2024-10-15)
------------------

Changed
^^^^^^^

- Modified OneHotEncodingTransformer, made an instance of OneHotEncoder and assign it to attribut _encoder `#308 <https://github.com/azukds/tubular/pull/309>`
- Refactored BaseDateTransformer, BaseDateTwoColumnTransformer and associated testing  `#273 <https://github.com/azukds/tubular/pull/273>`_
- BaseTwoColumnTransformer removed in favour of mixin classes TwoColumnMixin and NewColumnNameMixin to handle validation of two columns and new_column_name arguments `#273 <https://github.com/azukds/tubular/pull/273>`_
- Refactored tests for InteractionTransformer  `#283 <https://github.com/azukds/tubular/pull/283>`_
- Refactored tests for StringConcatenator and SeriesStrMethodTransformer, added separator mixin class. `#286 <https://github.com/azukds/tubular/pull/286>`_
- Refactored MeanResponseTransformer tests in new format `#262 <https://github.com/azukds/tubular/pull/262>`_
- refactored build tools and package config into pyproject.toml `#271 <https://github.com/azukds/tubular/pull/271>`_
- set up automatic versioning using setuptools-scm `#271 <https://github.com/azukds/tubular/pull/271>`_
- Refactored TwoColumnOperatorTransformer tests in new format `#274 <https://github.com/azukds/tubular/issues/274>`_
- Refactored PCATransformer tests in new format `#277 <https://github.com/azukds/tubular/issues/277>`_
- Refactored tests for NullIndicator `#301 <https://github.com/azukds/tubular/issues/301>`_
- Refactored BetweenDatesTransformer tests in new format `#294 <https://github.com/azukds/tubular/issues/294>`_
- As part of above, edited dates file transformers to use BaseDropOriginalMixin in transform
- Refactored DateDifferenceTransformer tests in new format. Had to turn off autodefine new_column_name functionality to match generic test expectations. Suggest we look to turn back on in the future. `#296 https://github.com/azukds/tubular/issues/296`
- Refactored DateDiffLeapYearTransformer tests in new format. As part of this had to remove the autodefined new_column_name, as this conflicts with the generic testing. Suggest we look to turn back on in future. `#295 https://github.com/azukds/tubular/issues/295`
- Edited base testing setup for dates file, created new BaseDatetimeTransformer class
- Refactored DatetimeInfoExtractor tests in new format `#297 <https://github.com/azukds/tubular/issues/297>`_
- Refactored DatetimeSinusoidCalculator tests in new format. `#310 <https://github.com/azukds/tubular/issues/310>`_
- fixed a bug in CappingTransformer which was preventing use of .get_params method `#311 <https://github.com/azukds/tubular/issues/311>`_
- Setup requirements for narwhals, remove python3.8 from our build pipelines as incompatible with polars
- Narwhal-ified BaseTransformer `#313 <https://github.com/azukds/tubular/issues/313>_`
- Refactored ToDatetimeTransformer tests in new format `#300 <https://github.com/azukds/tubular/issues/300>`_
- Refactors tests for SeriesDtMethodTransformer in new format. Changed column arg to columns to fit generic format. `#299 <https://github.com/azukds/tubular/issues/299>_`
- Refactored OrdinalEncoderTransformer tests in new format `#330 <https://github.com/azukds/tubular/issues/330>`_
- Narwhal-ified NullIndicator `#319 <https://github.com/azukds/tubular/issues/319>_`
- Narwhal-ified NearestMeanResponseImputer `#320 <https://github.com/azukds/tubular/issues/320>_`
- Narwhal-ified MedianImputer `#317 <https://github.com/azukds/tubular/issues/317>_`


1.3.1 (2024-07-18)
------------------
Changed
^^^^^^^

- Refactored NominalToIntegerTransformer tests in new format `#261 <https://github.com/azukds/tubular/pull/261>`_
- Refactored GroupRareLevelsTransformer tests in new format `#259 <https://github.com/azukds/tubular/pull/259>`_
- DatetimeInfoExtractor.mappings_provided changed from a dict.keys() object to list so transformer is serialisable. `#258 <https://github.com/azukds/tubular/pull/258>`_
- Created BaseNumericTransformer class to support test refactor of numeric file `#266 <https://github.com/azukds/tubular/pull/266>`_
- Updated testing approach for LogTransformer `#268 <https://github.com/azukds/tubular/pull/268>`_
- Refactored ScalingTransformer tests in new format `#284 <https://github.com/azukds/tubular/pull/284>`_


1.3.0 (2024-06-13)
------------------
Added
^^^^^
- Inheritable tests for generic base behaviours for base transformer in `base_tests.py`, with fixtures to allow for this in `conftest.py`
- Split existing input check into two better defined checks for TwoColumnOperatorTransformer `#183 <https://github.com/azukds/tubular/pull/183>`_
- Created unit tests for checking column type and size `#183 <https://github.com/azukds/tubular/pull/183>`_
- Automated weights column checks through a mixin class and captured common weight tests in generic test classes for weighted transformers

Changed
^^^^^^^
- Standardised naming of weight arg across transformers 
- Update DataFrameMethodTransformer tests to have inheritable init class that can be used by othe test files.
- Moved BaseTransformer, DataFrameMethodTransformer, BaseMappingTransformer, BaseMappingTransformerMixin, CrossColumnMappingTransformer and Mapping Transformer over to the new testing framework.
- Refactored MappingTransformer by removing redundant init method.
- Refactored tests for ColumnDtypeSetter, and renamed (from SetColumnDtype)
- Refactored tests for SetValueTransformer
- Refactored ArbitraryImputer by removing redundant fillna call in transform method. This should increase tubular's efficiency and maintainability.
- Fixed bugs in MedianImputer and ModeImputer where they would error for all null columns.
- Refactored ArbitraryImputer and BaseImputer tests in new format.
- Refactored MedianImputer tests in new format.
- Replaced occurrences of pd.Dataframe.drop() with del statement to speed up tubular. Note that no additional unit testing has been done for copy=False as this release is scheduled to remove copy. 
- Created BaseCrossColumnNumericTransformer class. Refactored CrossColumnAddTransformer and CrossColumnMultiplyTransformer to use this class. Moved tests for these objects to new approach.
- Created BaseCrossColumnMappingTransformer class and integrated into CrossColumnMappingTransformer tests  
- Refactored BaseNominalTransformer tests in new format & moved its logic to the transform method.
- Refactored ModeImputer tests in new format.
- Added generic init tests to base tests for transformers that take two columns as an input.
- Refactored EqualityChecker tests in new format.
- Bugfix to MeanResponseTransformer to ignore unobserved categorical levels
- Refactored dates.py to prepare for testing refactor. Edited BaseDateTransformer (and created BaseDateTwoColumnTransformer) to follow standard format, implementing validations at init/fit/transform. To reduce complexity of file, made transformers more opinionated to insist on specific and consistent column dtypes.  `#246 <https://github.com/azukds/tubular/pull/246>`_
- Added test_BaseTwoColumnTransformer base class for columns that require a list of two columns for input
- Added BaseDropOriginalMixin to mixin transformers to handle validation and method of dropping original features, also added appropriate test classes.
- Refactored MeanImputer tests in new format `#250 <https://github.com/azukds/tubular/pull/250>`_
- Refactored DatetimeInfoExtractor to condense and improve readability
- added minimal_dataframe_lookup fixture to conftest, and edited generic tests to use this
- Alphabetised the minimial attribute dictionary for readability.
- Refactored OHE transformer tests to align with new testing framework. 
- Moved fixtures relating only to a single test out of conftest and into testing script where utilised.
- !!!Introduced dependency on Sklearn's OneHotEncoder by adding test to check OHE transformer (which we are calling from within our OHE wrapper) is fit before transform 
- Refactored NearestMeanResponseImputer in line with new testing framework.


Removed
^^^^^^^
- Functionality for BaseTransformer (and thus all transformers) to take `None` as an option for columns. This behaviour was inconsistently implemented across transformers. Rather than extending to all we decided to remove this functionality. This required updating a lot of test files.
- The `columns_set_or_check()` method from BaseTransformer. With the above change it was no longer necessary. Subsequent updates to nominal transformers and their tests were required.
- Set pd copy_on_write to True (will become default in pandas 3.0) which allowed the functionality of the copy method of the transformers to be dropped `#197 <https://github.com/azukds/tubular/pull/197>`_

1.2.2 (2024-02-20)
------------------
Added
^^^^^
- Created unit test for checking if log1p is working and well conditioned for small x `#178 <https://github.com/azukds/tubular/pull/178>`_

Changed
^^^^^^^
- Changed LogTransformer to use log1p(x) instead of log(x+1) `#178 <https://github.com/azukds/tubular/pull/178>`_
- Changed unit tests using log(x+1) to log1p(x) `#178 <https://github.com/azukds/tubular/pull/178>`_

1.2.1 (2024-02-08)
------------------
Added
^^^^^
- Updated GroupRareLevelsTransformer so that when working with category dtypes it forgets categories encoded as rare (this is wanted behaviour as these categories are no longer present in the data) `#177 <https://github.com/azukds/tubular/pull/177>`_

1.2.0 (2024-02-06)
------------------
Added
^^^^^
- Update OneHotEncodingTransformer to default to returning int8 columns `#175 <https://github.com/azukds/tubular/pull/175>`_
- Updated NullIndicator to return int8 columns `#173 <https://github.com/azukds/tubular/pull/173>`_
- Updated MeanResponseTransformer to coerce return to float (useful behaviour for category type features) `#174 <https://github.com/azukds/tubular/pull/174>`_

1.1.1 (2024-01-18)
------------------

Added
^^^^^
- added type hints `#128 <https://github.com/azukds/tubular/pull/128>`_
- added some error handling to transform method of nominal transformers  `#162 <https://github.com/azukds/tubular/pull/162>`_
- added new release pipeline `#161 <https://github.com/azukds/tubular/pull/161>`_

1.1.0 (2023-12-19)
------------------

Added
^^^^^
- added flake8_bugbear (B) to ruff rules `#131 <https://github.com/azukds/tubular/pull/131>`_
- added flake8_datetimez (DTZ) to ruff rules `#132 <https://github.com/azukds/tubular/pull/132>`_
- added option to avoid passing unseen levels to rare in GroupRareLevelsTransformer `#141 <https://github.com/azukds/tubular/pull/141>`_

Changed
^^^^^^^
- minor changes to comply with flake8_bugbear (B) ruff rules `#131 <https://github.com/azukds/tubular/pull/131>`_
- minor changes to comply with flake8_datetimez (DTZ) ruff rules `#132 <https://github.com/azukds/tubular/pull/132>`_
- BaseMappingTransformerMixin chnaged to use Dataframe.replace rather than looping over columns `#135 <https://github.com/azukds/tubular/pull/135>`_
- MeanResponseTransformer.map_imputer_values() added to decouple from BaseMappingTransformerMixin `#135 <https://github.com/azukds/tubular/pull/135>`_
- BaseDateTransformer added to standardise datetime data handling `#148 <https://github.com/azukds/tubular/pull/148>`_

Removed
^^^^^^^
- removed some unnescessary implementation tests `#130 <https://github.com/azukds/tubular/pull/130>`_
- ReturnKeyDict class removed `#135 <https://github.com/azukds/tubular/pull/135>`_




1.0.0 (2023-07-24)
------------------

Changed
^^^^^^^
- now compatible with pandas>=2.0.0 `#123 <https://github.com/azukds/tubular/pull/123>`_
- DateDifferenceTransformer no longer supports 'Y' or  'M' units `#123 <https://github.com/azukds/tubular/pull/123>`_


0.3.8 (2023-07-10)
------------------

Changed
^^^^^^^
- replaced flake8 with ruff linting.  For a list of rules implemented, code changes made for compliance and further rule sets planned for future see PR  `#92 <https://github.com/azukds/tubular/pull/92>`_

0.3.7 (2023-07-05)
------------------

Changed
^^^^^^^
- minor change to `GroupRareLevelsTransformer` `test_super_transform_called` test to align with other cases `#90 <https://github.com/azukds/tubular/pull/90>`_
- removed pin of scikit-learn version to <1.20 `#90 <https://github.com/azukds/tubular/pull/90>`_
- update `black` version in pre-commit-config `#90 <https://github.com/azukds/tubular/pull/90>`_

0.3.6 (2023-05-24)
------------------

Added
^^^^^
- added support for vscode dev container with python 3.8, requirments-dev.txt, pylance/gitlens extensions and precommit all preinstalled `#83 <https://github.com/azukds/tubular/pull/83>`_

Changed
^^^^^^^
- added sklearn < 1.2 dependency `#86 <https://github.com/azukds/tubular/pull/86>`_

0.3.5 (2023-04-26)
------------------

Added
^^^^^
- added support for handling unseen levels in MeanResponseTransformer `#80 <https://github.com/azukds/tubular/pull/80>`_

Changed
^^^^^^^
- added pandas < 2.0.0 dependency `#81 <https://github.com/azukds/tubular/pull/81>`_

Deprecated
^^^^^^^^^^
- DateDifferenceTransformer M and Y units are incpompatible with pandas 2.0.0 and will be removed or changed in a future version `#81 <https://github.com/azukds/tubular/pull/81>`_

0.3.4 (2023-03-14)
------------------

Added
^^^^^
- added support for passing multiple columns and periods/units parameters to DatetimeSinusoidCalculator `#74 <https://github.com/azukds/tubular/pull/74>`_
- added support for handling a multi level response to MeanResponseTransformer `#67 <https://github.com/azukds/tubular/pull/67>`_

Changed
^^^^^^^
- changed ArbitraryImputer to preserve the dtype of columns (previously would upcast dtypes like int8 or float32) `#76 <https://github.com/azukds/tubular/pull/76>`_

Fixed
^^^^^

- fixed issue with OneHotencodingTransformer use of deprecated sklearn.OneHotEencoder.get_feature_names method `#66 <https://github.com/azukds/tubular/pull/66>`_

0.3.3 (2023-01-19)
------------------

Added
^^^^^
- added support for prior mean encoding (regularised encodings) `#46 <https://github.com/azukds/tubular/pull/46>`_

- added support for weights to mean, median and mode imputers `#47 <https://github.com/azukds/tubular/pull/47>`_

- added classname() method to BaseTransformer and prefixed all errors with classname call for easier debugging `#48 <https://github.com/azukds/tubular/pull/48>`_

- added DatetimeInfoExtractor transformer in ``tubular/dates.py`` associated tests with ``tests/dates/test_DatetimeInfoExtractor.py`` and examples with ``examples/dates/DatetimeInfoExtractor.ipynb`` `#49 <https://github.com/azukds/tubular/pull/49>`_

- added DatetimeSinusoidCalculator in ``tubular/dates.py`` associated tests with ``tests/dates/test_DatetimeSinusoidCalculator.py`` and examples with ``examples/dates/DatetimeSinusoidCalculator.ipynb`` `#50 <https://github.com/azukds/tubular/pull/50>`_

- added TwoColumnOperatorTransformer in ``tubular/numeric.py`` associated tests with ``tests/numeric/test_TwoColumnOperatorTransformer.py`` and examples with ``examples/dates/TwoColumnOperatorTransformer.ipynb`` `#51 <https://github.com/azukds/tubular/pull/51>`_

- added StringConcatenator in ``tubular/strings.py`` associated tests with ``tests/strings/test_StringConcatenator.py`` and examples with ``examples/strings/StringConcatenator.ipynb`` `#52 <https://github.com/azukds/tubular/pull/52>`_

- added SetColumnDtype in ``tubular/misc.py`` associated tests with ``tests/misc/test_StringConcatenator.py`` and examples with ``examples/strings/StringConcatenator.ipynb`` `#53 <https://github.com/azukds/tubular/pull/53>`_

- added warning to MappingTransformer in ``tubular/mapping.py`` for unexpected changes in dtype  `#54 <https://github.com/azukds/tubular/pull/54>`_

- added new module ``tubular/comparison.py`` containing EqualityChecker.  Also added associated tests with ``tests/comparison/test_EqualityChecker.py`` and examples with ``examples/comparison/EqualityChecker.ipynb`` `#55 <https://github.com/azukds/tubular/pull/55>`_

- added PCATransformer in ``tubular/numeric.py`` associated tests with ``tests/misc/test_PCATransformer.py`` and examples with ``examples/numeric/PCATransformer.ipynb`` `#57 <https://github.com/azukds/tubular/pull/57>`_

Fixed
^^^^^
- updated black version to 22.3.0 and flake8 version to 5.0.4 to fix compatibility issues `#45 <https://github.com/azukds/tubular/pull/45>`_

- removed kwargs argument from BaseTransfomer in ``tubular/base.py`` to avoid silent erroring if incorrect arguments passed to transformers. Fixed a few tests which were revealed to have incorrect arguments passed by change `#56 <https://github.com/azukds/tubular/pull/56>`_ 


0.3.2 (2022-01-13)
------------------

Added
^^^^^
- Added InteractionTransformer in ``tubular/numeric.py`` , associated tests with ``tests/numeric/test_InteractionTransformer.py`` file and examples with ``examples/numeric/InteractionTransformer.ipynb`` file.`#38 <https://github.com/azukds/tubular/pull/38>`_


0.3.1 (2021-11-09)
------------------

Added
^^^^^
- Added ``tests/test_transformers.py`` file with test to be applied all transformers `#30 <https://github.com/azukds/tubular/pull/30>`_

Changed
^^^^^^^
- Set min ``pandas`` version to 1.0.0 in ``requirements.txt``, ``requirements-dev.txt``, and ``docs/requirements.txt`` `#31 <https://github.com/azukds/tubular/pull/31>`_
- Changed ``y`` argument in fit to only accept ``pd.Series`` objects `#26 <https://github.com/azukds/tubular/pull/26>`_
- Added new ``_combine_X_y`` method to ``BaseTransformer`` which cbinds X and y `#26 <https://github.com/azukds/tubular/pull/26>`_
- Updated ``MeanResponseTransformer`` to use ``y`` arg in ``fit`` and remove setting ``response_column`` in init `#26 <https://github.com/azukds/tubular/pull/26>`_
- Updated ``OrdinalEncoderTransformer`` to use ``y`` arg in ``fit`` and remove setting ``response_column`` in init `#26 <https://github.com/azukds/tubular/pull/26>`_
- Updated ``NearestMeanResponseImputer`` to use ``y`` arg in ``fit`` and remove setting ``response_column`` in init `#26 <https://github.com/azukds/tubular/pull/26>`_
- Updated version of ``black`` used in the ``pre-commit-config`` to ``21.9b0`` `#25 <https://github.com/azukds/tubular/pull/25>`_
- Modified ``DataFrameMethodTransformer`` to add the possibility of drop original columns `#24 <https://github.com/azukds/tubular/pull/24>`_

Fixed
^^^^^
- Added attributes to date and numeric transformers to allow transformer to be printed `#30 <https://github.com/azukds/tubular/pull/30>`_
- Removed copy of mappings in ``MappingTransformer`` to allow transformer to work with sklearn.base.clone `#30 <https://github.com/azukds/tubular/pull/30>`_
- Changed data values used in some tests for ``MeanResponseTransformer`` so the test no longer depends on pandas <1.3.0 or >=1.3.0, required due to `change <https://pandas.pydata.org/docs/whatsnew/v1.3.0.html#float-result-for-groupby-mean-groupby-median-and-groupby-var>`_ `#25 <https://github.com/azukds/tubular/pull/25>`_  in pandas behaviour with groupby mean
- ``BaseTransformer`` now correctly raises ``TypeError`` exceptions instead of ``ValueError`` when input values are the wrong type `#26 <https://github.com/azukds/tubular/pull/26>`_
- Updated version of ``black`` used in the ``pre-commit-config`` to ``21.9b0`` `#25 <https://github.com/azukds/tubular/pull/25>`_

Removed
^^^^^^^
- Removed ``pytest`` and ``pytest-mock`` from ``requirements.txt`` `#31 <https://github.com/azukds/tubular/pull/31>`_

0.3.0 (2021-11-03)
------------------

Added
^^^^^
- Added ``scaler_kwargs`` as an empty attribute to the ``ScalingTransformer`` class to avoid an ``AttributeError`` raised by ``sklearn`` `#21 <https://github.com/azukds/tubular/pull/21>`_
- Added ``test-aide`` package to ``requirements-dev.txt`` `#21 <https://github.com/azukds/tubular/pull/21>`_
- Added logo for the package `#22 <https://github.com/azukds/tubular/pull/22>`_
- Added ``pre-commit`` to the project to manage pre-commit hooks `#22 <https://github.com/azukds/tubular/pull/22>`_
- Added `quick-start guide <https://tubular.readthedocs.io/en/latest/quick-start.html>`_ to docs `#22 <https://github.com/azukds/tubular/pull/22>`_
- Added `code of conduct <https://tubular.readthedocs.io/en/latest/code-of-conduct.html>`_ for the project `#22 <https://github.com/azukds/tubular/pull/22>`_

Changed
^^^^^^^
- Moved ``testing/test_data.py`` to ``tests`` folder `#21 <https://github.com/azukds/tubular/pull/21>`_
- Updated example notebooks to use California housing dataset from sklearn instead of Boston house prices dataset `#21 <https://github.com/azukds/tubular/pull/21>`_
- Changed ``changelog`` to be ``rst`` format and a changelog page added to docs `#22 <https://github.com/azukds/tubular/pull/22>`_
- Changed the default branch in the repository from ``master`` to ``main``

Removed
^^^^^^^
- Removed `testing` module and updated tests to use helpers from `test-aide` package `#21 <https://github.com/azukds/tubular/pull/21>`_

0.2.15 (2021-10-06)
-------------------

Added
^^^^^
- Add github action to run pytest, flake8, black and bandit `#10 <https://github.com/azukds/tubular/pull/10>`_

Changed
^^^^^^^
- Modified ``GroupRareLevelsTransformer`` to remove the constraint type of ``rare_level_name`` being string, instead it must be the same type as the columns selected `#13 <https://github.com/azukds/tubular/pull/13>`_
- Fix failing ``NullIndicator.transform`` tests `#14 <https://github.com/azukds/tubular/pull/14>`_

Removed
^^^^^^^
- Update ``NearestMeanResponseImputer`` to remove fallback to median imputation when no nulls present in a column `#10 <https://github.com/azukds/tubular/pull/10>`_

0.2.14 (2021-04-23)
-------------------

Added
^^^^^
- Open source release of the package on Github
