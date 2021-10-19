#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

std::string get_image_type(const cv::Mat& img, bool more_info=true) 
{
    std::string r;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');
   
    if (more_info)
        std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;

    return r;
}

void show_image(cv::Mat& img, std::string title)
{
    std::string image_type = get_image_type(img);
    cv::namedWindow(title + " type:" + image_type, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title, img);
    cv::waitKey(0);
}

auto DepthToCvImage(at::Tensor tensor, const uint8_t &bits = 1)
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

        show_image(output_mat, "converted image from tensor");
        return output_mat.clone();
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
    return cv::Mat(height, width, CV_8UC1);
}

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module> <path-to-input-image>\n";
    return -1;
  }

  std::cout<<"model to load: "<<argv[1]<<std::endl;
  at::init_num_threads();
  at::set_num_threads(1);

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  module.eval();
  std::cout << "ok\n";

  /** Read the image **/
  std::cout<<"reading input img: "<<argv[2]<<" ";
  cv::Mat img = cv::imread(argv[2]);
  std::cout<<"ok"<<std::endl;

  /** Resize image **/
  cv::resize(img, img, cv::Size(512, 384), CV_INTER_CUBIC);
  std::cout<<"img resize: "<<img.rows<<" x "<<img.cols<<std::endl;
  cv::imshow("Input image", img);
  cv::waitKey(0);
  auto tensor_image = torch::from_blob(img.data, {img.rows, img.cols, img.channels()}, at::kByte);
  tensor_image = tensor_image.permute({ 2,0,1 }); //[C x H x W]
  tensor_image.unsqueeze_(0); //[B x C x H x W]
  tensor_image = tensor_image.toType(c10::kFloat);//.sub(127.5).mul(0.0078125);
  tensor_image.to(c10::DeviceType::CPU);

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(tensor_image);

  // Execute the model and turn its output into a tensor.
  auto start = std::chrono::steady_clock::now();
  at::Tensor output = module.forward(inputs).toTensor();
  auto end = std::chrono::steady_clock::now();
  //std::cout<< output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  cv::Mat prediction = DepthToCvImage(output.squeeze());
  cv::imwrite("/tmp/midas_prediction.png", prediction);

  std::cout << "Elapsed time: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << " millisec"<<std::endl;

}
