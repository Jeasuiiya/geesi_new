add_test([=[TestDistributedIR.DeviceGraph]=]  /home/ai/ljj/tp/build/tests/ut/test_DistributedIR [==[--gtest_filter=TestDistributedIR.DeviceGraph]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TestDistributedIR.DeviceGraph]=]  PROPERTIES WORKING_DIRECTORY /home/ai/ljj/tp/build/tests/ut SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
add_test([=[TestDistributedIR.DivideGraph2SubGraph]=]  /home/ai/ljj/tp/build/tests/ut/test_DistributedIR [==[--gtest_filter=TestDistributedIR.DivideGraph2SubGraph]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TestDistributedIR.DivideGraph2SubGraph]=]  PROPERTIES WORKING_DIRECTORY /home/ai/ljj/tp/build/tests/ut SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_DistributedIR_TESTS TestDistributedIR.DeviceGraph TestDistributedIR.DivideGraph2SubGraph)
