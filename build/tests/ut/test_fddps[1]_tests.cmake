add_test([=[TestFDDPS.FDDPS]=]  /home/ai/ljj/tp/build/tests/ut/test_fddps [==[--gtest_filter=TestFDDPS.FDDPS]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TestFDDPS.FDDPS]=]  PROPERTIES WORKING_DIRECTORY /home/ai/ljj/tp/build/tests/ut SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_fddps_TESTS TestFDDPS.FDDPS)
