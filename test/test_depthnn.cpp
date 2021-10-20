#include <boost/test/unit_test.hpp>
#include <depthnn/DepthNN.hpp>

BOOST_AUTO_TEST_CASE(test_depth_map)
{
    cv::Mat img = cv::imread("test/data/zichao_apero.jpg", cv::IMREAD_COLOR);
    depthnn::DepthNN midas("/home/javi/rock/dev/install/models/midas_v21.pt");
    cv::Mat depthmap = midas.infer(img);
    cv::imwrite("test/data/out_zichao_depthmap.png", depthmap);

    BOOST_CHECK(img.cols == depthmap.cols);
    BOOST_CHECK(img.rows == depthmap.rows);
}
