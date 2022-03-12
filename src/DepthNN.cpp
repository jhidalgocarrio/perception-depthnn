#include "DepthNN.hpp"

#include <base-logging/Logging.hpp>
#include <sys/stat.h>

using namespace depthnn;


inline bool file_exist (const std::string& name)
{
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0); 
}

cv::Mat DepthToCvImage(at::Tensor tensor, const uint8_t &bits = 1)
{
    auto depth_min = tensor.min();
    auto depth_max = tensor.max();
    int height = tensor.sizes()[0];
    int width = tensor.sizes()[1];

    std::cout<<"height: "<<height<<" width: "<<width<<std::endl;
    std::cout<<"min depth: "<<depth_min<<std::endl;
    std::cout<<"max depth: "<<depth_max<<std::endl;

    int max_val = std::pow(2, (8*bits))-1;
    at::Tensor out = max_val * (tensor - depth_min) / (depth_max - depth_min);
    out = out.to(torch::kUInt8);
    std::cout<<"out min depth: "<<out.min()<<std::endl;
    std::cout<<"out max depth: "<<out.max()<<std::endl;

    try
    {
        cv::Mat output_mat (cv::Size{ width, height }, CV_8UC1, out.data_ptr<uint8_t>());

        std::cout<<"out image size: "<<output_mat.size()<<std::endl;
        return output_mat.clone();
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
    return cv::Mat(height, width, CV_8UC1);
}

DepthNN::DepthNN(const std::string &filename)
{
    /** Read model path **/
    if (!file_exist(filename))
    {
        LOG_ERROR_S << "[ERROR]: Given Torch model does not exist: "<<filename<<std::endl;
    }

    /* Load the model **/
    at::init_num_threads();
    at::set_num_threads(4);

    try {
        LOG_INFO_S<<"Model to load: "<<filename;
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(filename);
    }
    catch (const c10::Error& e)
    {
        LOG_FATAL_S<< "error loading the model\n";
    }

    module.eval();
    LOG_INFO_S<< "...[OK]\n";

}

cv::Mat DepthNN::infer(const cv::Mat &input)
{
    /** Resize the image **/
    cv::Mat img; cv::resize(input, img, cv::Size(512, 384), cv::INTER_CUBIC);

    /** Image to Tensor **/
    auto tensor_image = torch::from_blob(img.data, {img.rows, img.cols, img.channels()}, at::kByte);
    tensor_image = tensor_image.permute({ 2,0,1 }); //[C x H x W]
    tensor_image.unsqueeze_(0); //[B x C x H x W]
    tensor_image = tensor_image.toType(c10::kFloat);
    tensor_image.to(c10::DeviceType::CPU);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);

    /**  Execute the model and turn its output into a tensor. **/
    auto start = std::chrono::steady_clock::now();
    at::Tensor output = module.forward(inputs).toTensor();
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end-start);
    LOG_INFO_S<< "Inference time: "<<duration.count()<<" seconds\n";

    /** Convert prediction to image and resize to the original size **/
    cv::Mat prediction = DepthToCvImage(output.squeeze());
    cv::resize(prediction, prediction, cv::Size(input.cols, input.rows), cv::INTER_CUBIC);

    return prediction;
}


