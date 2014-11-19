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

bool loadFloatArray(std::vector<float>* p_outData,
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
		p_outData->resize(length);
		float* tArr = new float[length];
		is.read(reinterpret_cast<char*>(tArr),
			std::streamsize(length*sizeof(float))); // read data
		for (int i = 0; i < length; i++)
		{
			(*p_outData)[i] = tArr[i];
		}
		delete[] p_outData;
	}
	is.close();
	return true;
}


void saveFloatArrayPrompt(const float* p_inData, size_t length)
{
	string path = "../output/sav/debugDat.txt";
#ifndef _DEBUG
	OPENFILENAME ofn;
	char szFile[255];
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFile = szFile;
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "All\0*.*\0Text\0*.TXT\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
	bool hasFileName = GetSaveFileName(&ofn);
	if (hasFileName)
	{
		path = ofn.lpstrFile;
		saveFloatArray(p_inData, length, path);
	}
#else
	saveFloatArray(p_inData, length, path);
#endif
	//MessageBox(NULL, ofn.lpstrFile, "File Name", MB_OK);
}

void loadFloatArrayPrompt(std::vector<float>* p_outData)
{
	string path = "../output/sav/debugDat.txt";
#ifndef _DEBUG
	OPENFILENAME ofn;
	char szFile[255];
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFile = szFile;
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "All\0*.*\0Text\0*.TXT\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
	bool hasFileName = GetOpenFileName(&ofn);
	if (hasFileName)
	{
		path = ofn.lpstrFile;
		loadFloatArray(p_outData, path);
	}
#else
	loadFloatArray(p_outData, path);
#endif
	//MessageBox(NULL, ofn.lpstrFile, "File Name", MB_OK);
}