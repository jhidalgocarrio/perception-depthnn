#ifndef _DEPTH_NN_HPP_
#define _DEPTH_NN_HPP_

#include <opencv2/opencv.hpp>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include <string>

namespace depthnn
{
    class DepthNN
    {
    protected:
        torch::jit::script::Module module;
    public:
        /** @brief Default constructor **/
        DepthNN(const std::string &filename);

        cv::Mat infer(const cv::Mat &input);
    };

} // end namespace

#endif // _DEPTH_NN_HPP_