Changelog
=========

This changelog follows the great advice from https://keepachangelog.com/.

Each section will have a title of the format ``X.Y.Z (YYYY-MM-DD)`` giving the version of the package and the date of release of that version. Unreleased changes i.e. those that have been merged into `main` (e.g. with a .dev suffix) but which are not yet in a new release (on pypi) are added to the changelog but with the title ``X.Y.Z (unreleased)``. Unreleased sections can be combined when they are released and the date of release added to the title.

Subsections for each version can be one of the following;

- ``Added`` for new features.
- ``Changed`` for changes in existing functionality.
- ``Deprecated`` for soon-to-be removed features.
- ``Removed`` for now removed features.
- ``Fixed`` for any bug fixes.
- ``Security`` in case of vulnerabilities.

Each individual change should have a link to the pull request after the description of the change.

1.4.3 (unreleased)
------------------

Changed
^^^^^^^
- narwhalified ToDatetimeTransformer. Also made some usability improvements, e.g. to accept multiple columns `#379 <https://github.com/azukds/tubular/issues/379>_`
- fixed bug with MappingTransformer, BaseMappingTransformerMixin where nullable boolean mappings were being converted to non-nullable booleans
- Working on above, found additional bug with mapping null values. 
Considered removing this functionality, but it is actually needed for 
inverse pipelines. Changed this part of logic to work more like an imputer.
- placeholder 

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
- placeholder

1.4.2 (18/03/2025)
------------------

Changed
^^^^^^^

- converted OneHotEncodingTransformer to narwhals `#355 <https://github.com/lvgig/tubular/issues/355>_`
- updated WeightsColumnMixin to use new narwhals 'is_finite' method
- narwhalified ModeImputer `#321 <https://github.com/lvgig/tubular/issues/321>_`
- fixed issues with all null and nullable-bool column handling in dataframe_init_dispatch
- added NaN error handling to WeightColumnMixin
- narwhalified BaseNumericTransformer `#358 <https://github.com/lvgig/tubular/issues/358>_`
- narwhalified BaseCappingTransformer `#357 <https://github.com/lvgig/tubular/issues/357>_`
- narwhalified CappingTransformer `#361 <https://github.com/lvgig/tubular/issues/361>_`
- narwhalified OutOfRangeNullTransformer `#362 <https://github.com/lvgig/tubular/issues/362>_`
- narwhalified MeanImputer `#344 https://github.com/lvgig/tubular/issues/344_`
- narwhalified BaseGenericDateTransformer. As part of this updated test data handling of date columns
  across repo `#365 <https://github.com/lvgig/tubular/issues/365>_`
- narwhalified BaseNumericTransformer `#358 https://github.com/lvgig/tubular/issues/358`
- narwhalified DropOriginalMixin `#352 <https://github.com/lvgig/tubular/issues/352>_`
- narwhalified BaseMappingTransformer `#367 <https://github.com/lvgig/tubular/issues/367>_`
- narwhalified BaseMappingTransformerMixin. As part of this made mapping transformers more type-conscious, they now rely on an input 'return_dtypes' dict arg.`#369 <https://github.com/lvgig/tubular/issues/369>_`
- As part of #369, updated OrdinalEncoderTransformer to output Int8 type
- As part of #369, updated NominalToIntegerTransformer to output Int8 type. Removed inverse_mapping functionality, as this is more complicated when transform is opinionated on types.
- narwhalified GroupRareLevelsTransformer. As part of this, had to make transformer more opinionated and refuse columns with nulls (raises an error directing to imputers.) `#372 <https://github.com/lvgig/tubular/issues/372>_`
- narwhalified BaseDatetimeTransformer `#375 <https://github.com/azukds/tubular/issues/375>`
- Optional wanted_levels feature has been integrated into the OneHotEncodingTransformer which allows users to specify which levels in a column they wish to encode. `#384 <https://github.com/azukds/tubular/issues/384>_`
- Created unit tests to check if the values provided for wanted_values are as expected and if the output is as expected.
- fix: issue with falsey values not imputing for ArbitraryImputer `#391 <https://github.com/azukds/tubular/issues/391>_`

1.4.1 (02/12/2024)
------------------

Changed
^^^^^^^

- Refactored BaseImputer to utilise narwhals `#314 <https://github.com/lvgig/tubular/issues/314>_`
- Converted test dfs to flexible pandas/polars setup
- Converted BaseNominalTransformer to utilise narwhals `#334 <https://github.com/lvgig/tubular/issues/334>_`
- narwhalified CheckNumericMixin `#336 <https://github.com/lvgig/tubular/issues/336>_`
- Changed behaviour of NearestMeanResponseImputer so that if there are no nulls at fit, 
  it warns and has no effect at transform, as opposed to erroring. The error was problematic for e.g.
  lightweight test runs where nulls are less likely to be present.

1.4.0 (2024-10-15)
------------------

Changed
^^^^^^^

- Modified OneHotEncodingTransformer, made an instance of OneHotEncoder and assign it to attribut _encoder `#308 <https://github.com/lvgig/tubular/pull/309>`
- Refactored BaseDateTransformer, BaseDateTwoColumnTransformer and associated testing  `#273 <https://github.com/lvgig/tubular/pull/273>`_
- BaseTwoColumnTransformer removed in favour of mixin classes TwoColumnMixin and NewColumnNameMixin to handle validation of two columns and new_column_name arguments `#273 <https://github.com/lvgig/tubular/pull/273>`_
- Refactored tests for InteractionTransformer  `#283 <https://github.com/lvgig/tubular/pull/283>`_
- Refactored tests for StringConcatenator and SeriesStrMethodTransformer, added separator mixin class. `#286 <https://github.com/lvgig/tubular/pull/286>`_
- Refactored MeanResponseTransformer tests in new format `#262 <https://github.com/lvgig/tubular/pull/262>`_
- refactored build tools and package config into pyproject.toml `#271 <https://github.com/lvgig/tubular/pull/271>`_
- set up automatic versioning using setuptools-scm `#271 <https://github.com/lvgig/tubular/pull/271>`_
- Refactored TwoColumnOperatorTransformer tests in new format `#274 <https://github.com/lvgig/tubular/issues/274>`_
- Refactored PCATransformer tests in new format `#277 <https://github.com/lvgig/tubular/issues/277>`_
- Refactored tests for NullIndicator `#301 <https://github.com/lvgig/tubular/issues/301>`_
- Refactored BetweenDatesTransformer tests in new format `#294 <https://github.com/lvgig/tubular/issues/294>`_
- As part of above, edited dates file transformers to use BaseDropOriginalMixin in transform
- Refactored DateDifferenceTransformer tests in new format. Had to turn off autodefine new_column_name functionality to match generic test expectations. Suggest we look to turn back on in the future. `#296 https://github.com/lvgig/tubular/issues/296`
- Refactored DateDiffLeapYearTransformer tests in new format. As part of this had to remove the autodefined new_column_name, as this conflicts with the generic testing. Suggest we look to turn back on in future. `#295 https://github.com/lvgig/tubular/issues/295`
- Edited base testing setup for dates file, created new BaseDatetimeTransformer class
- Refactored DatetimeInfoExtractor tests in new format `#297 <https://github.com/lvgig/tubular/issues/297>`_
- Refactored DatetimeSinusoidCalculator tests in new format. `#310 <https://github.com/lvgig/tubular/issues/310>`_
- fixed a bug in CappingTransformer which was preventing use of .get_params method `#311 <https://github.com/lvgig/tubular/issues/311>`_
- Setup requirements for narwhals, remove python3.8 from our build pipelines as incompatible with polars
- Narwhal-ified BaseTransformer `#313 <https://github.com/lvgig/tubular/issues/313>_`
- Refactored ToDatetimeTransformer tests in new format `#300 <https://github.com/lvgig/tubular/issues/300>`_
- Refactors tests for SeriesDtMethodTransformer in new format. Changed column arg to columns to fit generic format. `#299 <https://github.com/lvgig/tubular/issues/299>_`
- Refactored OrdinalEncoderTransformer tests in new format `#330 <https://github.com/lvgig/tubular/issues/330>`_
- Narwhal-ified NullIndicator `#319 <https://github.com/lvgig/tubular/issues/319>_`
- Narwhal-ified NearestMeanResponseImputer `#320 <https://github.com/lvgig/tubular/issues/320>_`
- Narwhal-ified MedianImputer `#317 <https://github.com/lvgig/tubular/issues/317>_`


1.3.1 (2024-07-18)
------------------
Changed
^^^^^^^

- Refactored NominalToIntegerTransformer tests in new format `#261 <https://github.com/lvgig/tubular/pull/261>`_
- Refactored GroupRareLevelsTransformer tests in new format `#259 <https://github.com/lvgig/tubular/pull/259>`_
- DatetimeInfoExtractor.mappings_provided changed from a dict.keys() object to list so transformer is serialisable. `#258 <https://github.com/lvgig/tubular/pull/258>`_
- Created BaseNumericTransformer class to support test refactor of numeric file `#266 <https://github.com/lvgig/tubular/pull/266>`_
- Updated testing approach for LogTransformer `#268 <https://github.com/lvgig/tubular/pull/268>`_
- Refactored ScalingTransformer tests in new format `#284 <https://github.com/lvgig/tubular/pull/284>`_


1.3.0 (2024-06-13)
------------------
Added
^^^^^
- Inheritable tests for generic base behaviours for base transformer in `base_tests.py`, with fixtures to allow for this in `conftest.py`
- Split existing input check into two better defined checks for TwoColumnOperatorTransformer `#183 <https://github.com/lvgig/tubular/pull/183>`_
- Created unit tests for checking column type and size `#183 <https://github.com/lvgig/tubular/pull/183>`_
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
- Refactored dates.py to prepare for testing refactor. Edited BaseDateTransformer (and created BaseDateTwoColumnTransformer) to follow standard format, implementing validations at init/fit/transform. To reduce complexity of file, made transformers more opinionated to insist on specific and consistent column dtypes.  `#246 <https://github.com/lvgig/tubular/pull/246>`_
- Added test_BaseTwoColumnTransformer base class for columns that require a list of two columns for input
- Added BaseDropOriginalMixin to mixin transformers to handle validation and method of dropping original features, also added appropriate test classes.
- Refactored MeanImputer tests in new format `#250 <https://github.com/lvgig/tubular/pull/250>`_
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
- Set pd copy_on_write to True (will become default in pandas 3.0) which allowed the functionality of the copy method of the transformers to be dropped `#197 <https://github.com/lvgig/tubular/pull/197>`_

1.2.2 (2024-02-20)
------------------
Added
^^^^^
- Created unit test for checking if log1p is working and well conditioned for small x `#178 <https://github.com/lvgig/tubular/pull/178>`_

Changed
^^^^^^^
- Changed LogTransformer to use log1p(x) instead of log(x+1) `#178 <https://github.com/lvgig/tubular/pull/178>`_
- Changed unit tests using log(x+1) to log1p(x) `#178 <https://github.com/lvgig/tubular/pull/178>`_

1.2.1 (2024-02-08)
------------------
Added
^^^^^
- Updated GroupRareLevelsTransformer so that when working with category dtypes it forgets categories encoded as rare (this is wanted behaviour as these categories are no longer present in the data) `#177 <https://github.com/lvgig/tubular/pull/177>`_

1.2.0 (2024-02-06)
------------------
Added
^^^^^
- Update OneHotEncodingTransformer to default to returning int8 columns `#175 <https://github.com/lvgig/tubular/pull/175>`_
- Updated NullIndicator to return int8 columns `#173 <https://github.com/lvgig/tubular/pull/173>`_
- Updated MeanResponseTransformer to coerce return to float (useful behaviour for category type features) `#174 <https://github.com/lvgig/tubular/pull/174>`_

1.1.1 (2024-01-18)
------------------

Added
^^^^^
- added type hints `#128 <https://github.com/lvgig/tubular/pull/128>`_
- added some error handling to transform method of nominal transformers  `#162 <https://github.com/lvgig/tubular/pull/162>`_
- added new release pipeline `#161 <https://github.com/lvgig/tubular/pull/161>`_

1.1.0 (2023-12-19)
------------------

Added
^^^^^
- added flake8_bugbear (B) to ruff rules `#131 <https://github.com/lvgig/tubular/pull/131>`_
- added flake8_datetimez (DTZ) to ruff rules `#132 <https://github.com/lvgig/tubular/pull/132>`_
- added option to avoid passing unseen levels to rare in GroupRareLevelsTransformer `#141 <https://github.com/lvgig/tubular/pull/141>`_

Changed
^^^^^^^
- minor changes to comply with flake8_bugbear (B) ruff rules `#131 <https://github.com/lvgig/tubular/pull/131>`_
- minor changes to comply with flake8_datetimez (DTZ) ruff rules `#132 <https://github.com/lvgig/tubular/pull/132>`_
- BaseMappingTransformerMixin chnaged to use Dataframe.replace rather than looping over columns `#135 <https://github.com/lvgig/tubular/pull/135>`_
- MeanResponseTransformer.map_imputer_values() added to decouple from BaseMappingTransformerMixin `#135 <https://github.com/lvgig/tubular/pull/135>`_
- BaseDateTransformer added to standardise datetime data handling `#148 <https://github.com/lvgig/tubular/pull/148>`_

Removed
^^^^^^^
- removed some unnescessary implementation tests `#130 <https://github.com/lvgig/tubular/pull/130>`_
- ReturnKeyDict class removed `#135 <https://github.com/lvgig/tubular/pull/135>`_




1.0.0 (2023-07-24)
------------------

Changed
^^^^^^^
- now compatible with pandas>=2.0.0 `#123 <https://github.com/lvgig/tubular/pull/123>`_
- DateDifferenceTransformer no longer supports 'Y' or  'M' units `#123 <https://github.com/lvgig/tubular/pull/123>`_


0.3.8 (2023-07-10)
------------------

Changed
^^^^^^^
- replaced flake8 with ruff linting.  For a list of rules implemented, code changes made for compliance and further rule sets planned for future see PR  `#92 <https://github.com/lvgig/tubular/pull/92>`_

0.3.7 (2023-07-05)
------------------

Changed
^^^^^^^
- minor change to `GroupRareLevelsTransformer` `test_super_transform_called` test to align with other cases `#90 <https://github.com/lvgig/tubular/pull/90>`_
- removed pin of scikit-learn version to <1.20 `#90 <https://github.com/lvgig/tubular/pull/90>`_
- update `black` version in pre-commit-config `#90 <https://github.com/lvgig/tubular/pull/90>`_

0.3.6 (2023-05-24)
------------------

Added
^^^^^
- added support for vscode dev container with python 3.8, requirments-dev.txt, pylance/gitlens extensions and precommit all preinstalled `#83 <https://github.com/lvgig/tubular/pull/83>`_

Changed
^^^^^^^
- added sklearn < 1.2 dependency `#86 <https://github.com/lvgig/tubular/pull/86>`_

0.3.5 (2023-04-26)
------------------

Added
^^^^^
- added support for handling unseen levels in MeanResponseTransformer `#80 <https://github.com/lvgig/tubular/pull/80>`_

Changed
^^^^^^^
- added pandas < 2.0.0 dependency `#81 <https://github.com/lvgig/tubular/pull/81>`_

Deprecated
^^^^^^^^^^
- DateDifferenceTransformer M and Y units are incpompatible with pandas 2.0.0 and will be removed or changed in a future version `#81 <https://github.com/lvgig/tubular/pull/81>`_

0.3.4 (2023-03-14)
------------------

Added
^^^^^
- added support for passing multiple columns and periods/units parameters to DatetimeSinusoidCalculator `#74 <https://github.com/lvgig/tubular/pull/74>`_
- added support for handling a multi level response to MeanResponseTransformer `#67 <https://github.com/lvgig/tubular/pull/67>`_

Changed
^^^^^^^
- changed ArbitraryImputer to preserve the dtype of columns (previously would upcast dtypes like int8 or float32) `#76 <https://github.com/lvgig/tubular/pull/76>`_

Fixed
^^^^^

- fixed issue with OneHotencodingTransformer use of deprecated sklearn.OneHotEencoder.get_feature_names method `#66 <https://github.com/lvgig/tubular/pull/66>`_

0.3.3 (2023-01-19)
------------------

Added
^^^^^
- added support for prior mean encoding (regularised encodings) `#46 <https://github.com/lvgig/tubular/pull/46>`_

- added support for weights to mean, median and mode imputers `#47 <https://github.com/lvgig/tubular/pull/47>`_

- added classname() method to BaseTransformer and prefixed all errors with classname call for easier debugging `#48 <https://github.com/lvgig/tubular/pull/48>`_

- added DatetimeInfoExtractor transformer in ``tubular/dates.py`` associated tests with ``tests/dates/test_DatetimeInfoExtractor.py`` and examples with ``examples/dates/DatetimeInfoExtractor.ipynb`` `#49 <https://github.com/lvgig/tubular/pull/49>`_

- added DatetimeSinusoidCalculator in ``tubular/dates.py`` associated tests with ``tests/dates/test_DatetimeSinusoidCalculator.py`` and examples with ``examples/dates/DatetimeSinusoidCalculator.ipynb`` `#50 <https://github.com/lvgig/tubular/pull/50>`_

- added TwoColumnOperatorTransformer in ``tubular/numeric.py`` associated tests with ``tests/numeric/test_TwoColumnOperatorTransformer.py`` and examples with ``examples/dates/TwoColumnOperatorTransformer.ipynb`` `#51 <https://github.com/lvgig/tubular/pull/51>`_

- added StringConcatenator in ``tubular/strings.py`` associated tests with ``tests/strings/test_StringConcatenator.py`` and examples with ``examples/strings/StringConcatenator.ipynb`` `#52 <https://github.com/lvgig/tubular/pull/52>`_

- added SetColumnDtype in ``tubular/misc.py`` associated tests with ``tests/misc/test_StringConcatenator.py`` and examples with ``examples/strings/StringConcatenator.ipynb`` `#53 <https://github.com/lvgig/tubular/pull/53>`_

- added warning to MappingTransformer in ``tubular/mapping.py`` for unexpected changes in dtype  `#54 <https://github.com/lvgig/tubular/pull/54>`_

- added new module ``tubular/comparison.py`` containing EqualityChecker.  Also added associated tests with ``tests/comparison/test_EqualityChecker.py`` and examples with ``examples/comparison/EqualityChecker.ipynb`` `#55 <https://github.com/lvgig/tubular/pull/55>`_

- added PCATransformer in ``tubular/numeric.py`` associated tests with ``tests/misc/test_PCATransformer.py`` and examples with ``examples/numeric/PCATransformer.ipynb`` `#57 <https://github.com/lvgig/tubular/pull/57>`_

Fixed
^^^^^
- updated black version to 22.3.0 and flake8 version to 5.0.4 to fix compatibility issues `#45 <https://github.com/lvgig/tubular/pull/45>`_

- removed kwargs argument from BaseTransfomer in ``tubular/base.py`` to avoid silent erroring if incorrect arguments passed to transformers. Fixed a few tests which were revealed to have incorrect arguments passed by change `#56 <https://github.com/lvgig/tubular/pull/56>`_ 


0.3.2 (2022-01-13)
------------------

Added
^^^^^
- Added InteractionTransformer in ``tubular/numeric.py`` , associated tests with ``tests/numeric/test_InteractionTransformer.py`` file and examples with ``examples/numeric/InteractionTransformer.ipynb`` file.`#38 <https://github.com/lvgig/tubular/pull/38>`_


0.3.1 (2021-11-09)
------------------

Added
^^^^^
- Added ``tests/test_transformers.py`` file with test to be applied all transformers `#30 <https://github.com/lvgig/tubular/pull/30>`_

Changed
^^^^^^^
- Set min ``pandas`` version to 1.0.0 in ``requirements.txt``, ``requirements-dev.txt``, and ``docs/requirements.txt`` `#31 <https://github.com/lvgig/tubular/pull/31>`_
- Changed ``y`` argument in fit to only accept ``pd.Series`` objects `#26 <https://github.com/lvgig/tubular/pull/26>`_
- Added new ``_combine_X_y`` method to ``BaseTransformer`` which cbinds X and y `#26 <https://github.com/lvgig/tubular/pull/26>`_
- Updated ``MeanResponseTransformer`` to use ``y`` arg in ``fit`` and remove setting ``response_column`` in init `#26 <https://github.com/lvgig/tubular/pull/26>`_
- Updated ``OrdinalEncoderTransformer`` to use ``y`` arg in ``fit`` and remove setting ``response_column`` in init `#26 <https://github.com/lvgig/tubular/pull/26>`_
- Updated ``NearestMeanResponseImputer`` to use ``y`` arg in ``fit`` and remove setting ``response_column`` in init `#26 <https://github.com/lvgig/tubular/pull/26>`_
- Updated version of ``black`` used in the ``pre-commit-config`` to ``21.9b0`` `#25 <https://github.com/lvgig/tubular/pull/25>`_
- Modified ``DataFrameMethodTransformer`` to add the possibility of drop original columns `#24 <https://github.com/lvgig/tubular/pull/24>`_

Fixed
^^^^^
- Added attributes to date and numeric transformers to allow transformer to be printed `#30 <https://github.com/lvgig/tubular/pull/30>`_
- Removed copy of mappings in ``MappingTransformer`` to allow transformer to work with sklearn.base.clone `#30 <https://github.com/lvgig/tubular/pull/30>`_
- Changed data values used in some tests for ``MeanResponseTransformer`` so the test no longer depends on pandas <1.3.0 or >=1.3.0, required due to `change <https://pandas.pydata.org/docs/whatsnew/v1.3.0.html#float-result-for-groupby-mean-groupby-median-and-groupby-var>`_ `#25 <https://github.com/lvgig/tubular/pull/25>`_  in pandas behaviour with groupby mean
- ``BaseTransformer`` now correctly raises ``TypeError`` exceptions instead of ``ValueError`` when input values are the wrong type `#26 <https://github.com/lvgig/tubular/pull/26>`_
- Updated version of ``black`` used in the ``pre-commit-config`` to ``21.9b0`` `#25 <https://github.com/lvgig/tubular/pull/25>`_

Removed
^^^^^^^
- Removed ``pytest`` and ``pytest-mock`` from ``requirements.txt`` `#31 <https://github.com/lvgig/tubular/pull/31>`_

0.3.0 (2021-11-03)
------------------

Added
^^^^^
- Added ``scaler_kwargs`` as an empty attribute to the ``ScalingTransformer`` class to avoid an ``AttributeError`` raised by ``sklearn`` `#21 <https://github.com/lvgig/tubular/pull/21>`_
- Added ``test-aide`` package to ``requirements-dev.txt`` `#21 <https://github.com/lvgig/tubular/pull/21>`_
- Added logo for the package `#22 <https://github.com/lvgig/tubular/pull/22>`_
- Added ``pre-commit`` to the project to manage pre-commit hooks `#22 <https://github.com/lvgig/tubular/pull/22>`_
- Added `quick-start guide <https://tubular.readthedocs.io/en/latest/quick-start.html>`_ to docs `#22 <https://github.com/lvgig/tubular/pull/22>`_
- Added `code of conduct <https://tubular.readthedocs.io/en/latest/code-of-conduct.html>`_ for the project `#22 <https://github.com/lvgig/tubular/pull/22>`_

Changed
^^^^^^^
- Moved ``testing/test_data.py`` to ``tests`` folder `#21 <https://github.com/lvgig/tubular/pull/21>`_
- Updated example notebooks to use California housing dataset from sklearn instead of Boston house prices dataset `#21 <https://github.com/lvgig/tubular/pull/21>`_
- Changed ``changelog`` to be ``rst`` format and a changelog page added to docs `#22 <https://github.com/lvgig/tubular/pull/22>`_
- Changed the default branch in the repository from ``master`` to ``main``

Removed
^^^^^^^
- Removed `testing` module and updated tests to use helpers from `test-aide` package `#21 <https://github.com/lvgig/tubular/pull/21>`_

0.2.15 (2021-10-06)
-------------------

Added
^^^^^
- Add github action to run pytest, flake8, black and bandit `#10 <https://github.com/lvgig/tubular/pull/10>`_

Changed
^^^^^^^
- Modified ``GroupRareLevelsTransformer`` to remove the constraint type of ``rare_level_name`` being string, instead it must be the same type as the columns selected `#13 <https://github.com/lvgig/tubular/pull/13>`_
- Fix failing ``NullIndicator.transform`` tests `#14 <https://github.com/lvgig/tubular/pull/14>`_

Removed
^^^^^^^
- Update ``NearestMeanResponseImputer`` to remove fallback to median imputation when no nulls present in a column `#10 <https://github.com/lvgig/tubular/pull/10>`_

0.2.14 (2021-04-23)
-------------------

Added
^^^^^
- Open source release of the package on Github
