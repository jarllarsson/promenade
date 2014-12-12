#include "FileHandler.h"
#include <fstream>
#include "ToString.h"
#include "SettingsData.h"
#include "CurrentPathHelper.h"
#include <windows.h>
#include "StrTools.h"

bool write_file_binary(std::string const & filename,
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

bool saveFloatArray(std::vector<float>* p_inData,
	const std::string& file_path)
{
	std::ofstream os;
	os.open(file_path, std::ios::binary | std::ios::out);
	if (!os.good() || !os.is_open())
		return false;
	//os<<length; // write size
	size_t length = p_inData->size();
	os.write(reinterpret_cast<char*>(&(*p_inData)[0]),
		std::streamsize(length*sizeof(float))); // write data
	os.close();
	return true;
}

bool loadFloatArray(std::vector<float>* p_outData,
	const std::string& file_path)
{
	size_t bytelength = 0;
	std::ifstream is;
	is.open(file_path.c_str(), std::ios::binary | std::ios::in | std::ios::ate); // ate, place at end to read size
	if (!is.good() || !is.is_open())
		return false;
	//is>>length; // read size
	bytelength = is.tellg(); // byteLen
	if (bytelength > 0)
	{
		int members = bytelength / sizeof(float);
		p_outData->resize(members);
		float* tArr = new float[members]; // temp buffer array
		is.seekg(0, std::ios::beg); // place at start
		is.read(reinterpret_cast<char*>(tArr), bytelength); // read data to buffer
		for (int i = 0; i < members; i++) // copy to vector
		{
			(*p_outData)[i] = tArr[i];
		}
		delete[] tArr;
	}
	is.close();
	return true;
}



void saveFloatArrayPrompt(std::vector<float>* p_inData, int p_fileTypeIdx)
{
	std::string path = "../output/sav/biptest";
#ifndef _DEBUG
	OPENFILENAME ofn;
	char szFile[255];
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFile = szFile;
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "All\0*.*\0Biped Gait\0*.bgait\0Quadruped Gait\0*.qgait\0";
	ofn.nFilterIndex = p_fileTypeIdx;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
	bool hasFileName = GetSaveFileName(&ofn);
	if (hasFileName)
	{
		path = ofn.lpstrFile;
		saveFloatArray(p_inData, path);
	}
#else
	saveFloatArray(p_inData, path);
#endif
	//MessageBox(NULL, ofn.lpstrFile, "File Name", MB_OK);
}
void loadFloatArrayPrompt(std::vector<float>*& p_outData, int p_fileTypeIdx)
{
	std::string path = "../output/sav/biptest";
#ifndef _DEBUG
	OPENFILENAME ofn;
	char szFile[255];
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFile = szFile;
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "All\0*.*\0Biped Gait\0*.bgait\0Quadruped Gait\0*.qgait\0";
	ofn.nFilterIndex = p_fileTypeIdx;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
	bool hasFileName = GetOpenFileName(&ofn);
	if (hasFileName)
	{
		if (p_outData == NULL)
			p_outData = new std::vector<float>();
		path = ofn.lpstrFile;
		loadFloatArray(p_outData, path);
	}
#else
	if (p_outData == NULL)
		p_outData = new std::vector<float>();
	loadFloatArray(p_outData, path);
#endif
	//MessageBox(NULL, ofn.lpstrFile, "File Name", MB_OK);
}

bool writeSettings(SettingsData& p_settingsfile)
{
	std::string exePathPrefix = GetExecutablePathDirectory();
	std::string path = exePathPrefix + std::string("../settings.txt");
	std::vector<std::string> rows;
	// existing file
	std::ifstream is;
	is.open(path.c_str(), std::ios::in);
	if (!is.good() || !is.is_open())
		return false;
	//is>>length; // read size
	std::string tmpStr;
	// read all
	while (!is.eof())
	{
		std::getline(is, tmpStr);
		rows.push_back(tmpStr);
	}
	is.close();

	// go through each row, assume they are in order
	int optCounter = 0;
	for (int i = 0; i < rows.size(); i++)
	{
		if (!(rows[i][0] == '#' || rows[i] == ""))
		{
			// a writeable
			switch (optCounter)
			{
			case 0:
				rows[i] = p_settingsfile.m_fullscreen ? "1" : "0";
				break;
			case 1:
				rows[i] = p_settingsfile.m_appMode;
				break;
			case 2:
				rows[i] = ToString(p_settingsfile.m_wwidth);
				break;
			case 3:
				rows[i] = ToString(p_settingsfile.m_wheight);
				break;
			case 4:
				rows[i] = p_settingsfile.m_simMode;
				break;
			case 5:
				rows[i] = ToString(p_settingsfile.m_measurementRuns);
				break;
			case 6:
				rows[i] = p_settingsfile.m_pod;
				break;
			case 7:
				rows[i] = p_settingsfile.m_execMode;
				break;
			case 8:
				rows[i] = ToString(p_settingsfile.m_charcount_serial);
				break;
			case 9:
				rows[i] = ToString(p_settingsfile.m_parallel_invocs);
				break;
			case 10:
				rows[i] = ToString(p_settingsfile.m_charOffsetX);
				break;
			case 11:
				rows[i] = p_settingsfile.m_startPaused ? "1" : "0";
				break;
			case 12:
				rows[i] = ToString(p_settingsfile.m_optmesSteps);
				break;
			case 13:
				rows[i] = ToString(p_settingsfile.m_optW_fd);
				break;
			case 14:
				rows[i] = ToString(p_settingsfile.m_optW_fv);
				break;
			case 15:
				rows[i] = ToString(p_settingsfile.m_optW_fh);
				break;
			case 16:
				rows[i] = ToString(p_settingsfile.m_optW_fr);
				break;
			case 17:
				rows[i] = ToString(p_settingsfile.m_optW_fp);
				break;
			default:
				// do nothing
				break;
			}
			optCounter++;
		}
	}

	// resave, using altered rows structure
	std::ofstream os;
	os.open(path, std::ios::out);
	if (!os.good() || !os.is_open())
		return false;

	for (int i = 0; i < rows.size(); i++)
	{
		os << rows[i] << "\n";
	}
	os.close();
}

bool loadSettings(SettingsData& p_settingsfile)
{
	std::string exePathPrefix = GetExecutablePathDirectory();
	std::string path = exePathPrefix + std::string("../settings.txt");
	std::ifstream is;
	is.open(path.c_str(), std::ios::in);
	if (!is.good() || !is.is_open())
		return false;
	//is>>length; // read size
	std::string tmpStr;
	int tmpInt = 0;
	float tmpFlt = 0.0f;
	auto readDiscardHeader = [](std::ifstream* p_is)->void // explicit return type "->void"
	{
		std::string stmp = "x";
		do { std::getline(*p_is, stmp); } while (stmp == "");
	};
	// Fullscreen
	std::getline(is, tmpStr); // throwaway title
	is >> tmpInt;
	p_settingsfile.m_fullscreen = tmpInt == 0 ? false : true;
	// app mode
	readDiscardHeader(&is);
	std::getline(is, tmpStr);
	p_settingsfile.m_appMode = tmpStr[0];
	// window width
	readDiscardHeader(&is);
	is >> tmpInt;
	p_settingsfile.m_wwidth = tmpInt;
	// window height
	readDiscardHeader(&is);
	is >> tmpInt;
	p_settingsfile.m_wheight = tmpInt;
	// sim mode
	readDiscardHeader(&is);
	std::getline(is, tmpStr);
	p_settingsfile.m_simMode = tmpStr[0];
	// measurement runs
	readDiscardHeader(&is);
	is >> tmpInt;
	p_settingsfile.m_measurementRuns = tmpInt;
	// pod
	readDiscardHeader(&is);
	std::getline(is, tmpStr);
	p_settingsfile.m_pod = tmpStr[0];
	// exec mode
	readDiscardHeader(&is);
	std::getline(is, tmpStr);
	p_settingsfile.m_execMode = tmpStr[0];
	// charcount serial
	readDiscardHeader(&is);
	is >> tmpInt;
	p_settingsfile.m_charcount_serial = tmpInt;
	// parallel invocs
	readDiscardHeader(&is);
	is >> tmpInt;
	p_settingsfile.m_parallel_invocs = tmpInt;
	// char offset X
	readDiscardHeader(&is);
	is >> tmpFlt;
	p_settingsfile.m_charOffsetX = tmpFlt;
	// start paused
	readDiscardHeader(&is);
	is >> tmpInt;
	p_settingsfile.m_startPaused = tmpInt == 0 ? false : true;
	// op steaps
	readDiscardHeader(&is);
	is >> tmpInt;
	p_settingsfile.m_optmesSteps = tmpInt;
	// optimization weights
	readDiscardHeader(&is);
	std::getline(is, tmpStr); // extra header here
	is >> tmpFlt; p_settingsfile.m_optW_fd = tmpFlt;
	readDiscardHeader(&is);
	is >> tmpFlt; p_settingsfile.m_optW_fv = tmpFlt;
	readDiscardHeader(&is);
	is >> tmpFlt; p_settingsfile.m_optW_fh = tmpFlt;
	readDiscardHeader(&is);
	is >> tmpFlt; p_settingsfile.m_optW_fr = tmpFlt;
	readDiscardHeader(&is);
	is >> tmpFlt; p_settingsfile.m_optW_fp = tmpFlt;

	is.close();
	return true;
}

bool saveMeasurementToCollectionFileAtRow(std::string& p_filePath, float p_average, float p_std, int p_rowIdx)
{
	std::string exePathPrefix = GetExecutablePathDirectory();
	std::string path = exePathPrefix + p_filePath;
	std::vector<std::string> rows;
	// existing file
	std::ifstream is;
	is.open(path.c_str(), std::ios::in);
	if (!is.good() || !is.is_open())
		return false;
	//is>>length; // read size
	std::string tmpStr;
	// read all
	while (!is.eof())
	{
		std::getline(is, tmpStr);
		if (tmpStr!="") rows.push_back(tmpStr);
	}
	is.close();

	// Find the row with the right index (the first character denotes the index)
	int vectorIdx = -1;
	int lastIdx = 0;
	for (int i = 0; i < rows.size(); i++)
	{
		std::string firstChars = rows[i].substr(0, rows[i].find(" "));
		if (firstChars != "#" && firstChars != "")
		{
			int idx = stringToInt(firstChars);
			lastIdx = idx;
			if (p_rowIdx == idx)
			{
				vectorIdx = i;
				break;
			}
		}
	}
	if (vectorIdx == -1) // not found
	{
		vectorIdx=rows.size()-1;
		while (lastIdx<p_rowIdx)
		{
			lastIdx++;
			vectorIdx++;
			rows.push_back(ToString(lastIdx) + " 0 0 0 0");
		}
	}

	float ylow = p_average - p_std, yhigh = p_average + p_std;
	std::string newRow = ToString(p_rowIdx) + " " + ToString(p_average) + " " + ToString(p_std) + " " + ToString(ylow) + " " + ToString(yhigh);
	// replace
	rows[vectorIdx] = newRow;


	// now write
	std::ofstream os;
	os.open(path, std::ios::out);
	if (!os.good() || !os.is_open())
		return false;

	for (int i = 0; i < rows.size(); i++)
	{
		os << rows[i]<<"\n";
	}

	//os<<length; // write size
	//size_t length = p_inData->size();
	//os.write(reinterpret_cast<char*>(&(*p_inData)[0]),
	//	std::streamsize(length*sizeof(float))); // write data
	os.close();
}
