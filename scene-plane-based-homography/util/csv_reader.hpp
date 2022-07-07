#include <string>
#include <fstream>
#include <vector>
#include <sstream>

namespace util {

// \class: CSVReader
// read csv file and get data.
class CSVReader {
  std::string filename_;
  std::string delim_;

 public:
  CSVReader(std::string filename, std::string delim=",")
      : filename_(filename), delim_(delim)
  {}

  // get data from csv file using stringstream.
  std::vector<std::vector<std::string>> GetData() {
    std::ifstream file(filename_);
    std::vector<std::vector<std::string>> data;

    std::string line="";

    while(std::getline(file, line)) {
      std::vector<std::string> vec;

      std::stringstream ss(line);
      std::string substr;

      while(ss.good()) {
        std::getline(ss, substr, ',');
        vec.push_back(substr);
      }
      data.push_back(vec);

    }
    file.close();

    return data;
  }
};

}  // namespace util
