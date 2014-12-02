#pragma once
#include <vector>
#include <string>
#include <fstream>

using namespace std;

// =======================================================================================
//                                  Measurement Bin
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # MeasurementBin
/// 
/// 6-4-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

template<class T>
class MeasurementBin
{
public:
	MeasurementBin();
	void activate();
	float calculateMean();
	float calculateSTD();
	void finishRound();
	bool saveResultsCSV(const string& p_fileName);
	bool saveResultsGNUPLOT(const string& p_fileName);
	void saveMeasurement(T p_measurement);
	void saveMeasurement(T p_measurement, float p_timeStamp);
	void saveMeasurement(T p_measurement, int p_timeStamp);
	void saveMeasurementRelTStamp(T p_measurement, float p_deltaTimeStamp);
	bool isActive();
private:
	vector<T> m_measurements;
	vector<float> m_timestamps;
	double m_mean;
	double m_std;
	vector<double> m_allMeans;
	vector<double> m_allSTDs;
	bool m_active;
};

template<class T>
MeasurementBin<T>::MeasurementBin()
{
	m_active = false;
	m_mean = 0.0;
	m_std = 0.0f;
}

template<class T>
float MeasurementBin<T>::calculateMean()
{
	return 0.0f;
}

template<class T>
float MeasurementBin<T>::calculateSTD()
{
	return 0.0f;
}


template<>
float MeasurementBin<float>::calculateMean();

template<>
float MeasurementBin<float>::calculateSTD();

template<class T>
bool MeasurementBin<T>::saveResultsCSV(const string& p_fileName)
{
	if (m_active)
	{
		ofstream outFile;
		string file = p_fileName + ".csv";
		outFile.open(file);

		if (!outFile.good())
		{
			return false;
		}
		else
		{
			// Gfx settings
			if (m_allMeans.size() > 0 && m_allSTDs.size() > 0 &&
				m_allMeans.size() == m_allSTDs.size())
			{
				outFile << "Mean time,Standard deviation" << "\n";
				for (int i = 0; i < m_allMeans.size(); i++)
				{
					outFile << m_allMeans[i] << "," << m_allSTDs[i] << "\n";
				}
			}
			outFile << "\nRaw measurements\n";
			if (m_timestamps.size() == m_measurements.size())
			{
				outFile << "\nTimestamp,Measurement\n";
				for (int i = 0; i < m_measurements.size(); i++)
				{
					outFile << m_timestamps[i] << "," << m_measurements[i] << "\n";
				}
			}
			else
			{
				for (int i = 0; i < m_measurements.size(); i++)
				{
					outFile << m_measurements[i] << "\n";
				}
			}

		}
		outFile.close();
		return true;
	}
	return false;
}

template<class T>
bool MeasurementBin<T>::saveResultsGNUPLOT(const string& p_fileName)
{
	if (m_active)
	{
		ofstream outFile;
		string file = p_fileName + ".gnuplot.txt";
		outFile.open(file);

		if (!outFile.good())
		{
			return false;
		}
		else
		{
			outFile << "# " << p_fileName << "\n";
			// if means and STD exist, print them
			if (m_allMeans.size() > 0 && m_allSTDs.size() > 0 &&
				m_allMeans.size() == m_allSTDs.size())
			{			
				outFile << "# step - mean - standard deviation" << "\n";
				if (m_timestamps.size() == m_allMeans.size())
				{
					for (int i = 0; i < m_allMeans.size(); i++)
					{
						outFile << i << " " << m_allMeans[i] << " " << m_allSTDs[i] << " # " << m_timestamps[i] << "\n";
					}
				}
				else
				{
					for (int i = 0; i < m_allMeans.size(); i++)
					{
						outFile << i << " " << m_allMeans[i] << " " << m_allSTDs[i] <<  "\n";
					}
				}
			}
			else if (m_timestamps.size() == m_measurements.size())// else print the raw measurements
			{
				outFile << "# step - measurement" << "\n";
				for (int i = 0; i < m_allMeans.size(); i++)
				{
					outFile << i << " " << m_allMeans[i] << " " << m_allSTDs[i] << " # " << m_timestamps[i] << "\n";
				}
			}
			else
			{
				outFile << "# step - measurement" << "\n";
				for (int i = 0; i < m_allMeans.size(); i++)
				{
					outFile << i << " " << m_allMeans[i] << " " << m_allSTDs[i] << "\n";
				}
			}
		}
		outFile.close();
		return true;
	}
	return false;
}

template<class T>
void MeasurementBin<T>::finishRound()
{
	if (m_active)
	{
		calculateMean();
		calculateSTD();
		m_allMeans.push_back(m_mean);
		m_allSTDs.push_back(m_std);
	}
}

template<class T>
void MeasurementBin<T>::activate()
{
	m_active = true;
}

template<class T>
void MeasurementBin<T>::saveMeasurement(T p_measurement, float p_timeStamp)
{
	if (m_active)
	{
		m_measurements.push_back(p_measurement);
		m_timestamps.push_back(p_timeStamp);
	}

}

template<class T>
void MeasurementBin<T>::saveMeasurement(T p_measurement, int p_timeStamp)
{
	if (m_active)
	{
		m_measurements.push_back(p_measurement);
		m_timestamps.push_back((float)p_timeStamp);
	}

}

template<class T>
void MeasurementBin<T>::saveMeasurement(T p_measurement)
{
	if (m_active)
	{
		m_measurements.push_back(p_measurement);
	}

}

template<class T>
bool MeasurementBin<T>::isActive()
{
	return m_active;
}

template<class T>
void MeasurementBin<T>::saveMeasurementRelTStamp(T p_measurement, float p_deltaTimeStamp)
{
	if (m_active)
	{
		float oldTimeStamp = 0.0f;
		if (m_timestamps.size() > 0) oldTimeStamp = m_timestamps.back();
		m_measurements.push_back(p_measurement);
		m_timestamps.push_back(p_deltaTimeStamp+oldTimeStamp);
	}
}
