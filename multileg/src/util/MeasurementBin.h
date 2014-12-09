#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "CurrentPathHelper.h"

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
	void accumulateMeasurementAt(float p_measurement, int p_idx); // for doing several runs and storing multiple values on same spot
	void saveMeasurementRelTStamp(T p_measurement, float p_deltaTimeStamp);
	bool isActive();
	double getMean();
	double getSTD();
private:
	vector<T> m_measurements;
	vector<float> m_timestamps;
	double m_mean;
	double m_std;
	vector<double> m_allMeans;
	vector<double> m_allSTDs;
	bool m_active;
	int m_internalRuns;
};

template<class T>
double MeasurementBin<T>::getSTD()
{
	return m_std;
}

template<class T>
double MeasurementBin<T>::getMean()
{
	return m_mean;
}

template<class T>
MeasurementBin<T>::MeasurementBin()
{
	m_active = false;
	m_mean = 0.0;
	m_std = 0.0f;
	m_internalRuns = 0; // only used if T is vector
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

template<>
float MeasurementBin<std::vector<float>>::calculateMean();

template<>
float MeasurementBin<std::vector<float>>::calculateSTD();



template<>
bool MeasurementBin<float>::saveResultsCSV(const string& p_fileName);

template<>
bool MeasurementBin<float>::saveResultsGNUPLOT(const string& p_fileName);

template<>
bool MeasurementBin<std::vector<float>>::saveResultsCSV(const string& p_fileName);

template<>
bool MeasurementBin<std::vector<float>>::saveResultsGNUPLOT(const string& p_fileName);



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

template<>
void MeasurementBin<std::vector<float>>::finishRound();



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



template<>
void MeasurementBin<std::vector<float>>::accumulateMeasurementAt(float p_measurement, int p_idx);


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
