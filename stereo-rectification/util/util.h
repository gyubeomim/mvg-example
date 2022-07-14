#ifndef UTIL_H_
#define UTIL_H_

#include <string>

#define PROJECT_PATH "/mnt/d/edward/gitrepo/mvg-example/stereo-rectification"

namespace util {

std::string GetProjectPath() {
  return PROJECT_PATH;
}

}  // namespace util

#endif /* UTIL_H_ */
