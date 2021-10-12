#include <boost/test/unit_test.hpp>
#include <depthnn/Dummy.hpp>

using namespace depthnn;

BOOST_AUTO_TEST_CASE(it_should_not_crash_when_welcome_is_called)
{
    depthnn::DummyClass dummy;
    dummy.welcome();
}
