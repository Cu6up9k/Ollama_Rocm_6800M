#include "sycl_intel_gpu_mgr.hpp"

bool is_supported_gpu(device dev) {
  return supported_gpus.count(dev.get_info<ext::intel::info::device::device_id>()) != 0;
}

std::vector<device> get_supported_gpu_devices() {
  std::vector<device> devices;
  for (auto const& this_platform : platform::get_platforms()) {
    if (this_platform.get_info<info::platform::name>() == "Intel(R) Level-Zero") {
      for (auto& dev : this_platform.get_devices()) {
        if (dev.is_gpu() && is_supported_gpu(dev)) {
          devices.push_back(dev);
        }
      }
    }
  }
  return devices;
}

int get_device_num() { return get_supported_gpu_devices().size(); }

void get_dev_info(int dev_idx, gpu_info* info) {
  auto gpu_devices = get_supported_gpu_devices();
  if (dev_idx < gpu_devices.size() && dev_idx >= 0) {
    auto dev = gpu_devices[dev_idx];
    std::strcpy(info->dev.vendor_name, dev.get_info<info::device::vendor>().c_str());
    std::strcpy(info->dev.device_name, dev.get_info<info::device::name>().c_str());
    std::strcpy(info->runtime.driver_version, dev.get_info<info::device::driver_version>().c_str());
    info->dev.device_id = dev.get_info<ext::intel::info::device::device_id>();
    info->runtime.free_mem = dev.get_info<ext::intel::info::device::free_memory>();
    info->runtime.global_mem_size = dev.get_info<info::device::global_mem_size>();
  }
}
