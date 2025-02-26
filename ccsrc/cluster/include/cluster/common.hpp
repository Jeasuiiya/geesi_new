#include <string>
#include <vector>

class ServiceNode {
    std::string name;
    std::vector<ServiceNode> node;
    std::vector<Device> devices;
    long totalMemory;
    std::vector<Link> links;
} class Device {
    int type;
    std::string name;
    long memory;
} class Link {
    int type;
    std::vector<Device&> devices;
    long totalBandwidth;
}
