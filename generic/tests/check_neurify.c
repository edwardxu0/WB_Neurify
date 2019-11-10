#include "check_nnet.c"
#include "check_split.c"

Suite *neurify_suite()
{
    Suite *suite = suite_create("neurify");
    return suite;
}

int main()
{
    SRunner *runner = srunner_create(neurify_suite());

    srunner_add_suite(runner, nnet_suite());
    srunner_add_suite(runner, split_suite());

    // srunner_run_all(runner, CK_VERBOSE);
    srunner_run_all(runner, CK_NORMAL);
    int ntests_failed = srunner_ntests_failed(runner);
    srunner_free(runner);

    return (ntests_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}