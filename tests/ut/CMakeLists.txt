add_cc_gtest(test_DistributedIR SRCS test_DistributedIR.cc DEPENDS DistributedIR)

add_cc_gtest(test_Common SRCS test_common.cc DEPENDS CommonHeaders DistributedIR)

add_cc_gtest(test_fddps SRCS test_fddps.cc DEPENDS FDDPSPolicy)
