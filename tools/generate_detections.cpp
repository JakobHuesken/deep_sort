#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

using namespace std;
using namespace cv;
using namespace tensorflow;

void run_in_batches(std::function<void(const std::vector<Tensor> &)> f, const std::vector<Tensor> &data_x, std::vector<Tensor> &out, int batch_size) {
    int data_len = out.size();
    int num_batches = data_len / batch_size;

    int s = 0, e = 0;
    for (int i = 0; i < num_batches; i++) {
        s = i * batch_size;
        e = (i + 1) * batch_size;
        std::vector<Tensor> batch_data_x(data_x.begin() + s, data_x.begin() + e);
        f(batch_data_x);
        for (int j = s; j < e; j++) {
            out[j] = batch_data_x[j - s];
        }
    }
    if (e < data_len) {
        std::vector<Tensor> batch_data_x(data_x.begin() + e, data_x.end());
        f(batch_data_x);
        for (int j = e; j < data_len; j++) {
            out[j] = batch_data_x[j - e];
        }
    }
}

cv::Mat extract_image_patch(const cv::Mat &image, const cv::Rect &bbox, const cv::Size &patch_shape) {
    cv::Rect patched_bbox = bbox;

    if (patch_shape.width > 0 && patch_shape.height > 0) {
        double target_aspect = static_cast<double>(patch_shape.width) / patch_shape.height;
        double new_width = target_aspect * patched_bbox.height;
        patched_bbox.x -= (new_width - patched_bbox.width) / 2;
        patched_bbox.width = new_width;
    }

    // Convert to top left, bottom right
    patched_bbox.width += patched_bbox.x;
    patched_bbox.height += patched_bbox.y;

    // Clip at image boundaries
    patched_bbox.x = std::max(0, patched_bbox.x);
    patched_bbox.y = std::max(0, patched_bbox.y);
    patched_bbox.width = std::min(image.cols - 1, patched_bbox.width);
    patched_bbox.height = std::min(image.rows - 1, patched_bbox.height);

    if (patched_bbox.x >= patched_bbox.width || patched_bbox.y >= patched_bbox.height) {
        return cv::Mat();
    }

    cv::Mat image_patch = image(patched_bbox);
    cv::resize(image_patch, image_patch, patch_shape);
    return image_patch;
}

class ImageEncoder {
public:
    ImageEncoder(const std::string &checkpoint_filename, const std::string &input_name = "images", const std::string &output_name = "features") {
        SessionOptions session_options;
        session_options.config.mutable_gpu_options()->set_allow_growth(true);

        Session *session;
        Status status = NewSession(session_options, &session);
        if (!status.ok()) {
            cerr << "Failed to create TensorFlow session: " << status.ToString() << endl;
            exit(1);
        }

        GraphDef graph_def;
        status = ReadBinaryProto(Env::Default(), checkpoint_filename, &graph_def);
        if (!status.ok()) {
            cerr << "Failed to read model file: " << status.ToString() << endl;
            exit(1);
        }

        status = session->Create(graph_def);
        if (!status.ok()) {
            cerr << "Failed to load model: " << status.ToString() << endl;
            exit(1);
        }

        session_ = session;
        input_var_ = input_name + ":0";
        output_var_ = output_name + ":0";

        assert(session_->Run({}, {input_var_}, {}, &input_tensor_).ok());
        assert(session_->Run({}, {output_var_}, {}, &output_tensor_).ok());

        feature_dim_ = output_tensor_.shape().dim_size(1);
        image_shape_ = cv::Size(input_tensor_.shape().dim_size(2), input_tensor_.shape().dim_size(1));
    }

    std::vector<float> operator()(const cv::Mat &data_x, int batch_size = 32) {
        std::vector<Tensor> out(data_x.rows, Tensor(DT_FLOAT, TensorShape({1, feature_dim_})));
        std::vector<Tensor> input_tensors;

        for (int i = 0; i < data_x.rows; i++) {
            cv::Mat image_patch = data_x.row(i);
            cv::cvtColor(image_patch, image_patch, cv::COLOR_BGR2RGB);
            Tensor input_tensor(DT_UINT8, TensorShape({1, image_patch.rows, image_patch.cols, 3}));
            auto input_tensor_mapped = input_tensor.tensor<uint8_t, 4>();
            for (int c = 0; c < 3; c++) {
                for (int y = 0; y < image_patch.rows; y++) {
                    for (int x = 0; x < image_patch.cols; x++) {
                        input_tensor_mapped(0, y, x, c) = image_patch.at<cv::Vec3b>(y, x)[c];
                    }
                }
            }
            input_tensors.push_back(std::move(input_tensor));
        }

        run_in_batches([this](const std::vector<Tensor> &input_tensors) {
            Session *session = session_;
            std::vector<Tensor> output_tensors;
            Status status = session->Run({{input_var_, input_tensors}}, {output_var_}, {}, &output_tensors);
            if (!status.ok()) {
                cerr << "Failed to run model: " << status.ToString() << endl;
                exit(1);
            }
            for (int i = 0; i < output_tensors.size(); i++) {
                const auto &output_tensor = output_tensors[i];
                float *output_data = out[i].flat<float>().data();
                const float *data = output_tensor.flat<float>().data();
                std::memcpy(output_data, data, feature_dim_ * sizeof(float));
            }
        }, input_tensors, out, batch_size);

        std::vector<float> output_data;
        for (const Tensor &tensor : out) {
            const float *data = tensor.flat<float>().data();
            output_data.insert(output_data.end(), data, data + feature_dim_);
        }
        return output_data;
    }

    private:
        Session *session_;
        std::string input_var_;
        std::string output_var_;
        int feature_dim_;
        cv::Size image_shape_;
        Tensor input_tensor_;
        Tensor output_tensor_;
    };

std::function<std::vector<float>(const cv::Mat &, const std::vector<cv::Rect> &)> create_box_encoder(const std::string &model_filename, const std::string &input_name = "images", const std::string &output_name = "features", int batch_size = 32) {
    ImageEncoder image_encoder(model_filename, input_name, output_name);

    return [image_encoder](const cv::Mat &image, const std::vector<cv::Rect> &boxes) -> std::vector<float> {
        std::vector<cv::Mat> image_patches;
        for (const cv::Rect &box : boxes) {
            cv::Mat patch = extract_image_patch(image, box, image_encoder.image_shape());
            if (patch.empty()) {
                cerr << "WARNING: Failed to extract image patch: " << box << "." << endl;
                patch = cv::Mat(image_encoder.image_shape(), CV_8UC3, cv::Scalar(0, 0, 0));
            }
            image_patches.push_back(std::move(patch));
        }

        cv::Mat image_patches_mat(image_patches.size(), image_encoder.image_shape().width, image_encoder.image_shape().height, CV_8UC3);
        for (int i = 0; i < image_patches.size(); i++) {
            image_patches[i].copyTo(image_patches_mat.row(i));
        }

        std::vector<float> features = image_encoder(image_patches_mat, batch_size);
        return features;
    };
}

void generate_detections(std::function<std::vector<float>(const cv::Mat &, const std::vector<cv::Rect> &)> encoder, const std::string &mot_dir, const std::string &output_dir, const std::string &detection_dir = "") {
    //check if detection dir exists, otherwise use the mot dir
    if (detection_dir.empty()) {
        detection_dir = mot_dir;
    }
    //create the output dir
    if (mkdir(output_dir.c_str(), 0755) != 0) {
        if (errno != EEXIST) {
            cerr << "Failed to create output directory '" << output_dir << "'" << endl;
            exit(1);
        }
    }

    for (const string &sequence : fs::directory_iterator(mot_dir)) { //iterate through the directories
        cout << "Processing " << sequence << endl;
        const string sequence_dir = mot_dir + "/" + sequence;
        const string image_dir = sequence_dir + "/img1";

        std::unordered_map<int, string> image_filenames;
        for (const string &filename : fs::directory_iterator(image_dir)) { //iterate through the images
            int frame_number = std::stoi(filename.substr(0, filename.find('.')));
            image_filenames[frame_number] = image_dir + "/" + filename;
        }

        const string detection_file = detection_dir + "/" + sequence + "/det/det.txt";
        vector<vector<float>> detections_in;
        ifstream detection_stream(detection_file);
        if (detection_stream.is_open()) {
            string line;
            while (getline(detection_stream, line)) {
                istringstream line_stream(line);
                vector<float> detection(10);
                for (int i = 0; i < 10; i++) {
                    line_stream >> detection[i];
                }
                detections_in.push_back(detection);
            }
        }

        vector<vector<float>> detections_out;

        vector<int> frame_indices;
        for (const vector<float> &detection : detections_in) {
            frame_indices.push_back(static_cast<int>(detection[0]));
        }
        int min_frame_idx = *min_element(frame_indices.begin(), frame_indices.end());
        int max_frame_idx = *max_element(frame_indices.begin(), frame_indices.end());

        for (int frame_idx = min_frame_idx; frame_idx <= max_frame_idx; frame_idx++) {
            cout << "Frame " << setw(5) << setfill('0') << frame_idx << "/" << setw(5) << setfill('0') << max_frame_idx << endl;

            vector<vector<float>> rows;
            for (const vector<float> &detection : detections_in) {
                if (static_cast<int>(detection[0]) == frame_idx) {
                    rows.push_back(detection);
                }
            }

            if (image_filenames.find(frame_idx) == image_filenames.end()) {
                cout << "WARNING: could not find image for frame " << frame_idx << endl;
                continue;
            }

            cv::Mat bgr_image = cv::imread(image_filenames[frame_idx], cv::IMREAD_COLOR);
            vector<cv::Rect> boxes;
            for (const vector<float> &row : rows) {
                boxes.push_back(cv::Rect(static_cast<int>(row[2]), static_cast<int>(row[3]), static_cast<int>(row[4]), static_cast<int>(row[5])));
            }
            vector<float> features = encoder(bgr_image, boxes);

            for (int i = 0; i < rows.size(); i++) {
                rows[i].insert(rows[i].end(), features.begin() + i * encoder(bgr_image, boxes).size(), features.begin() + (i + 1) * encoder(bgr_image, boxes).size());
                detections_out.push_back(rows[i]);
            }
        }

        const string output_filename = output_dir + "/" + sequence + ".npy";
        ofstream output_stream(output_filename, ios::binary);
        for (const vector<float> &detection : detections_out) {
            for (const float &value : detection) {
                output_stream.write(reinterpret_cast<const char*>(&value), sizeof(float));
            }
        }
    }
}

int main() {
    const string model_filename = "resources/networks/mars-small128.pb";
    const string mot_dir = "path/to/MOTChallenge";
    const string output_dir = "detections";
    const string detection_dir = "";  // Set to custom detection directory if needed

    auto encoder = create_box_encoder(model_filename, "images", "features", 32);
    generate_detections(encoder, mot_dir, output_dir, detection_dir);

    return 0;
}
