/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <fstream>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/geesibling/graph.pb.h"
#include "tensorflow/core/geesibling/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::tstring;

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name, std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status = ReadTextOrBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

namespace tf = tensorflow;
int main(int argc, char* argv[]) {
    // These are the command-line flags the program can understand.
    // They define where the graph and input data is located, and what kind of
    // input the model expects. If you train your own model, or use something
    // other than inception_v3, then you'll need to update these.
    string graph = "tensorflow/core/ir/placement/examples/data/regression.pbtxt";
    string loss_layer = "loss/loss";
    string train_op = "train_op";
    string output_layer = "loss/add";
    string init_layer = "init_all_vars_op";
    bool self_test = false;
    string root_dir = "";
    int epoch = 100;
    std::vector<Flag> flag_list = {
        Flag("graph", &graph, "graph to be executed"),
        Flag("init_layer", &init_layer, "name of init"),
        Flag("train_op", &train_op, "name of train op"),
        Flag("output_layer", &output_layer, "name of output layer"),
        Flag("epoch", &epoch, "epoch"),
    };
    string usage = tensorflow::Flags::Usage(argv[0], flag_list);
    const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
    if (!parse_result) {
        LOG(ERROR) << usage;
        return -1;
    }

    // We need to call this to set up global state for TensorFlow.
    tensorflow::port::InitMain(argv[0], &argc, &argv);
    if (argc > 1) {
        LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
        return -1;
    }
    tf::SessionOptions session_options;
    session_options.config.mutable_gpu_options()->set_allow_growth(true);
    // First we load and initialize the model.
    std::unique_ptr<tensorflow::Session> session(tf::NewSession(session_options));
    string graph_path = tensorflow::io::JoinPath(root_dir, graph);
    Status load_graph_status = LoadGraph(graph_path, &session);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        return -1;
    }

    // Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape{2, 1});
    // float* x_data = x.flat<float>().data();
    // x_data[0] = 1.0;
    // x_data[1] = 2.0;
    // std::vector<std::pair<string, Tensor>> inputs = {{input_x_layer, x}};

    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({}, {}, {init_layer}, nullptr);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }
    for (int e = 0; e < epoch; e++) {
        run_status = session->Run({}, {}, {train_op}, &outputs);
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed: " << run_status;
            return -1;
        } else {
            for (auto i : outputs) {
                for (int j = 0; j < i.flat<float>().size(); j++) {
                    LOG(INFO) << i.flat<float>().data()[j];
                }
            }
        }
    }

    return 0;
}