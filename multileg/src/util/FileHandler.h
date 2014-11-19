#pragma once
#include <fstream>
#include <string>
#include "ToString.h"

bool write_file_binary (std::string const & filename, 
						char const * data, size_t const bytes)
{
	std::ofstream b_stream(filename.c_str(), 
		std::fstream::out | std::fstream::binary);
	if (b_stream)
	{
		b_stream.write(data, bytes);
		return (b_stream.good());
	}
	return false;
}

bool saveFloatArray(const float* p_inData, size_t length, 
	const std::string& file_path)
{
	std::ofstream os;
	os.open(file_path, std::ios::binary | std::ios::out);
	if (!os.good() || !os.is_open())
		return false;
	os<<length; // write size
	os.write(reinterpret_cast<const char*>(p_inData), 
		std::streamsize(length*sizeof(float))); // write data
	os.close();
	return true;
}

bool loadFloatArray(float* p_outData, 
	const std::string& file_path)
{
	size_t length = 0;
	std::ifstream is;
	is.open(file_path.c_str(), std::ios::binary | std::ios::in);
	if (!is.good() || !is.is_open())
		return false;
	is>>length; // read size
	if (length>0)
	{
		is.read(reinterpret_cast<char*>(p_outData),
			std::streamsize(length*sizeof(float))); // read data
	}
	is.close();
	return true;
}