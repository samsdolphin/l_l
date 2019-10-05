#ifndef __COMMON_TOOLS_H__
#define __COMMON_TOOLS_H__

#include "logger.hpp"
#include "timer.hpp"
#include "os_compatible.hpp"
#include <future>

namespace COMMON_TOOLS
{
	template <typename T>
	static  std::vector<T>  vector_2d_to_1d(std::vector<std::vector<T>> & pt_vec_vec)
	{
		std::vector<T> pt_vec;
		for (unsigned int i = 0; i < pt_vec_vec.size(); i++)
			pt_vec.insert(pt_vec.end(), pt_vec_vec[i].begin(), pt_vec_vec[i].end());
		return pt_vec;
	};

	template <typename T>
	void maintain_maximum_thread_pool(std::list<T> &thread_pool, size_t maximum_parallel_thread)
	{
		if(thread_pool.size() >= maximum_parallel_thread)
		{
			while (1)
			{
				for (auto it = thread_pool.begin(); it != thread_pool.end(); it++)
				{
					auto status = (*it)->wait_for(std::chrono::nanoseconds(1));
					if (status == std::future_status::ready)
					{
						delete *it;
						thread_pool.erase(it);
						break;
					}
				}
				if(thread_pool.size() < maximum_parallel_thread)
					break;
				std::this_thread::sleep_for(std::chrono::nanoseconds(1));
			}
		}
	};

	inline bool if_file_exit(const std::string &name)
	{
		struct stat buffer; //Copy from: https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
		return (stat(name.c_str(), &buffer) == 0);
	};
}
#endif
