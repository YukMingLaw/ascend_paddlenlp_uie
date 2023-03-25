import numpy as np
import acl
import traceback
import struct

# error code
ACL_ERROR_NONE = 0
# rule for mem
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEM_MALLOC_HUGE_ONLY = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2
# rule for memory copy
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3

def check_ret(message, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret={}"
                        .format(message, ret))

buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index
    }

class AscendEngine(object):
    def __init__(self, device_id, max_inputs_shape, max_outputs_shape, model_path):
        acl.init('')
        self._is_destroyed = False
        self.device_id = device_id
        self.model_path = model_path
        self.model_id = None
        self.context = None
        self.input_data = []
        self.output_data = []
        self.output_data_host = []
        self.model_desc = None
        self.load_input_dataset = None
        self.load_output_dataset = None

        self.input_node_size = 0
        self.output_node_size = 0

        self.input_info = []
        self.output_info = []

        self.max_inputs_shape = max_inputs_shape
        self.max_outputs_shape = max_outputs_shape

        self.init_resource()

    def destroy(self):
        if self._is_destroyed:
            return

        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)
        # acl.rt.destroy_context(self.context)
        acl.finalize()

        self._is_destroyed = True

    def __del__(self):
        self.destroy()

    def init_resource(self):
        acl.rt.set_device(self.device_id)
        #self.context, ret = acl.rt.create_context(self.device_id)
        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        self.model_desc = acl.mdl.create_desc()
        self._get_model_info()
        print("init resource success")

    def _get_model_info(self,):
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        self.input_node_size = acl.mdl.get_num_inputs(self.model_desc)
        self.output_node_size = acl.mdl.get_num_outputs(self.model_desc)
        for i in range(self.input_node_size):
            dims, ret = acl.mdl.get_input_dims(self.model_desc, i)
            data_type = acl.mdl.get_input_data_type(self.model_desc, i)
            self.input_info.append({'dims_info': dims, 'data_type': data_type})
            print("input node[{}] info:{}".format(i, dims))
        for i in range(self.output_node_size):
            dims, ret = acl.mdl.get_output_dims(self.model_desc, i)
            data_type = acl.mdl.get_output_data_type(self.model_desc, i)
            self.output_info.append({'dims_info': dims, 'data_type': data_type})
            print("output node[{}] info:{}".format(i, dims))
        # malloc data buf in device memory
        self._gen_data_buffer()

    def _gen_data_buffer(self):
        for i in range(self.input_node_size):
            temp_buffer_size = int(np.prod(self.max_inputs_shape[i]) * np.dtype(self._trans_AclType_to_Dtype(self.input_info[i]['data_type'])).itemsize)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)
            self.input_data.append({"buffer": temp_buffer, "size": temp_buffer_size})
        
        for i in range(self.output_node_size):
            temp_buffer_size = int(np.prod(self.max_outputs_shape[i]) * np.dtype(self._trans_AclType_to_Dtype(self.output_info[i]['data_type'])).itemsize)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)
            self.output_data.append({"buffer": temp_buffer, "size": temp_buffer_size})
            temp, ret = acl.rt.malloc_host(temp_buffer_size)
            if ret != 0:
                raise Exception("can't malloc_host ret={}".format(ret))
            self.output_data_host.append({"size": temp_buffer_size, "buffer": temp})


    def _data_interaction(self, dataset, policy=ACL_MEMCPY_HOST_TO_DEVICE):
        temp_data_buffer = self.input_data if policy == ACL_MEMCPY_HOST_TO_DEVICE else self.output_data

        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            for i in range(len(self.output_data_host)):
                dataset.append(self.output_data_host[i])

        for i, item in enumerate(temp_data_buffer):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                bytes_in = dataset[i].tobytes()
                ptr = acl.util.bytes_to_ptr(bytes_in)
                ret = acl.rt.memcpy(item["buffer"],
                                    int(np.prod(self.input_info[i]['dims_info']['dims']) * np.dtype(self._trans_AclType_to_Dtype(self.input_info[i]['data_type'])).itemsize),
                                    ptr,
                                    int(np.prod(self.input_info[i]['dims_info']['dims']) * np.dtype(self._trans_AclType_to_Dtype(self.input_info[i]['data_type'])).itemsize),
                                    policy)
                check_ret("acl.rt.memcpy", ret)

            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr,
                                    int(np.prod(self.output_info[i]['dims_info']['dims']) * np.dtype(self._trans_AclType_to_Dtype(self.output_info[i]['data_type'])).itemsize),
                                    item["buffer"],
                                    int(np.prod(self.output_info[i]['dims_info']['dims']) * np.dtype(self._trans_AclType_to_Dtype(self.output_info[i]['data_type'])).itemsize),
                                    policy)
                check_ret("acl.rt.memcpy", ret)

    def _gen_dataset(self, type_str="input"):
        dataset = acl.mdl.create_dataset()

        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data
        for i, item in enumerate(temp_dataset):
            data = acl.create_data_buffer(item["buffer"], item["size"])
            if data is None:
                ret = acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)
            _, ret = acl.mdl.add_dataset_buffer(dataset, data)

            if type_str == "in":
                tensor_desc = acl.create_tensor_desc(-1, self.input_info[i]['dims_info']['dims'], -1)
                dataset, _ = acl.mdl.set_dataset_tensor_desc(dataset, tensor_desc, i)
            if ret != ACL_ERROR_NONE:
                ret = acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, images):
        print("data interaction from host to device")
        for i in range(self.input_node_size):
            self.input_info[i]['dims_info']['dims'] = list(images[i].shape)
        # copy images to device
        self._data_interaction(images, ACL_MEMCPY_HOST_TO_DEVICE)
        print("data interaction from host to device success")

    def _data_from_device_to_host(self):
        print("data interaction from device to host")
        for i in range(self.output_node_size):
            tensorDesc =acl.mdl.get_dataset_tensor_desc(self.load_output_dataset, i)
            dim_num = acl.get_tensor_desc_num_dims(tensorDesc)
            temp_shape = []
            for d in range(dim_num):
                dim_size, ret = acl.get_tensor_desc_dim_v2(tensorDesc, d)
                temp_shape.append(dim_size)
            # self.output_info[i]['dims_info']['dims'] = temp_shape
        res = []
        # copy device to host
        self._data_interaction(res, ACL_MEMCPY_DEVICE_TO_HOST)
        print("data interaction from device to host success")
        result = self.get_result(res)
        self._destroy_databuffer()
        return result

    def run(self, images):
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], np.ndarray):
            self._data_from_host_to_device(images)
            self._gen_dataset('in')
            self._gen_dataset("out")
        else:
            print('images not a known type')
            return
        self.forward()
        return self._data_from_device_to_host()

    def forward(self):
        ret = acl.mdl.execute(self.model_id,
                          self.load_input_dataset,
                          self.load_output_dataset)
        check_ret("acl.mdl.execute", ret)
        print('model inference success')

    def benchmark(self):
        begin_time = time.time()
        self.forward()
        end_time = time.time()
        print('model inference time:', end_time - begin_time)

    def _destroy_databuffer(self):
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)

    
    def _trans_AclType_to_Dtype(self, type):
        if type == -1:  # ACL_DT_UNDEFINED
            return -1
        elif type == 0:  # ACL_FLOAT
            return np.float32
        elif type == 1:
            return np.float16
        elif type == 2:
            return np.int8
        elif type == 3:
            return np.int32
        elif type == 4:
            return np.uint8
        elif type == 6:
            return np.int16
        elif type == 7:
            return np.uint16
        elif type == 8:
            return np.uint32
        elif type == 9:
            return np.int64
        elif type == 10:
            return np.uint64
        elif type == 11:
            return np.float64
        elif type == 12:
            return np.bool

    def get_result(self, output_data):
        dataset = []
        for i, temp in enumerate(output_data):
            size = int(np.prod(self.output_info[i]['dims_info']['dims']) * np.dtype(self._trans_AclType_to_Dtype(self.output_info[i]['data_type'])).itemsize)
            ptr = temp["buffer"]
            bytes_out = acl.util.ptr_to_bytes(ptr, size)
            data = np.frombuffer(bytes_out, dtype=self._trans_AclType_to_Dtype(self.output_info[i]['data_type'])).reshape(self.output_info[i]['dims_info']['dims'])
            dataset.append(data)
        return dataset