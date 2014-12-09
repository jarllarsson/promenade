#include "MeasurementBin.h"

template<>
void MeasurementBin<std::vector<float>>::finishRound()
{
	if (m_active)
	{
		calculateMean();
		calculateSTD();
	}
}

template<>
float MeasurementBin<float>::calculateMean()
{
	double accumulate = 0.0;
	unsigned int count = (unsigned int)m_measurements.size();
	for (unsigned int i = 0; i < count; i++)
	{
		accumulate += (double)m_measurements[i];
	}
	m_mean = accumulate / (double)count;
	return (float)m_mean;
}



template<>
float MeasurementBin<float>::calculateSTD()
{
	unsigned int count = (unsigned int)m_measurements.size();
	double squaredDistsToMean = 0.0;
	for (unsigned int i = 0; i < count; i++)
	{
		double dist = (double)m_measurements[i] - (double)m_mean;
		squaredDistsToMean += dist*dist;
	}
	double standardDeviation = sqrt(squaredDistsToMean / (double)count);
	m_std = standardDeviation;
	return (float)standardDeviation;
}

template<>
float MeasurementBin<std::vector<float>>::calculateMean()
{
	unsigned int count = (unsigned int)m_measurements.size();
	unsigned int totcounted = 0;
	m_allMeans.clear();
	for (unsigned int i = 0; i < count; i++)
	{
		double accumulate = 0.0;
		unsigned int runs = (unsigned int)m_measurements[i].size();
		m_internalRuns = runs;
		for (unsigned int j = 0; j < runs; j++)
		{
			accumulate += (double)m_measurements[i][j];
			totcounted++;
		}
		double thisMean = accumulate / (double)runs;
		m_allMeans.push_back(thisMean);
		m_mean += accumulate;
	}
	m_mean /= (double)totcounted;
	return m_mean; // return the total mean

}

template<>
float MeasurementBin<std::vector<float>>::calculateSTD()
{
	float mean = calculateMean();
	unsigned int count = (unsigned int)m_allMeans.size();
	double squaredDistsToMean = 0.0;
	unsigned int totcounted = 0;
	for (unsigned int i = 0; i < count; i++)
	{
		double localMean = m_allMeans[i];
		double localSquaredDistsToMean = 0.0;
		unsigned int runs = (unsigned int)m_measurements[i].size();
		for (unsigned int j = 0; j < runs; j++)
		{
			double dist = (double)m_measurements[i][j] - (double)localMean;
			double globDist = (double)m_measurements[i][j] - (double)mean;
			localSquaredDistsToMean += dist*dist;
			squaredDistsToMean += globDist*globDist;
			totcounted++;
		}
		double localStandardDeviation = sqrt(localSquaredDistsToMean / (double)runs);
		m_allSTDs.push_back(localStandardDeviation);
	}
	double standardDeviation = sqrt(squaredDistsToMean / (double)totcounted);
	m_std = standardDeviation;
	return (float)standardDeviation;
}

template<>
void MeasurementBin<std::vector<float>>::accumulateMeasurementAt(float p_measurement, int p_idx)
{
	if (m_active)
	{
		if (p_idx >= m_measurements.size())
		{
			m_measurements.push_back(std::vector<float>());
		}
		if (p_idx < m_measurements.size())
			m_measurements[p_idx].push_back(p_measurement);
	}
}

template<>
bool MeasurementBin<float>::saveResultsCSV(const string& p_fileName)
{
	if (m_active)
	{
		ofstream outFile;
		string file = GetExecutablePathDirectory() + p_fileName + ".csv";
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
				outFile << "Mean time (" << m_mean << "),Standard deviation (" << m_std << ")" << "\n";
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

template<>
bool MeasurementBin<float>::saveResultsGNUPLOT(const string& p_fileName)
{
	if (m_active)
	{
		ofstream outFile;
		string file = GetExecutablePathDirectory() + p_fileName + ".gnuplot.txt";
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
				outFile << "# step - mean (" << m_mean << ") - standard deviation (" << m_std << ")" << "\n";
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
						outFile << i << " " << m_allMeans[i] << " " << m_allSTDs[i] << "\n";
					}
				}
			}
			else if (m_timestamps.size() == m_measurements.size())// else print the raw measurements
			{
				outFile << "# step - measurement - (timestamp)" << "\n";
				for (int i = 0; i < m_measurements.size(); i++)
				{
					outFile << i << " " << m_measurements[i] << " # " << m_timestamps[i] << "\n";
				}
			}
			else
			{
				outFile << "# step - measurement" << "\n";
				for (int i = 0; i < m_measurements.size(); i++)
				{
					outFile << i << " " << m_measurements[i] << "\n";
				}
			}
		}
		outFile.close();
		return true;
	}
	return false;
}


template<>
bool MeasurementBin<std::vector<float>>::saveResultsCSV(const string& p_fileName)
{
	if (m_active)
	{
		ofstream outFile;
		string file = GetExecutablePathDirectory() + p_fileName + ".csv";
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
				outFile << "Mean time (" << m_mean << "),Standard deviation (" << m_std << ") r=" << m_internalRuns << "\n";
				for (int i = 0; i < m_allMeans.size(); i++)
				{
					outFile << m_allMeans[i] << "," << m_allSTDs[i] << "\n";
				}
			}
		}
		outFile.close();
		return true;
	}
	return false;
}

template<>
bool MeasurementBin<std::vector<float>>::saveResultsGNUPLOT(const string& p_fileName)
{
	if (m_active)
	{
		ofstream outFile;
		string file = GetExecutablePathDirectory() + p_fileName + ".gnuplot.txt";
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
				outFile << "# step - mean (" << m_mean << ") - standard deviation (" << m_std << ") r=" << m_internalRuns << " - ylow - yhigh\n";
				if (m_timestamps.size() == m_allMeans.size())
				{
					for (int i = 0; i < m_allMeans.size(); i++)
					{
						outFile << i << " " << m_allMeans[i] << " " << m_allSTDs[i] << " "<< m_allMeans[i] - m_allSTDs[i] << " " << m_allMeans[i] + m_allSTDs[i] << " # " << m_timestamps[i] << "\n";
					}
				}
				else
				{
					for (int i = 0; i < m_allMeans.size(); i++)
					{
						outFile << i << " " << m_allMeans[i] << " " << m_allSTDs[i] << " " << m_allMeans[i] - m_allSTDs[i] << " " << m_allMeans[i] + m_allSTDs[i]  << "\n";
					}
				}
			}
		}
		outFile.close();
		return true;
	}
	return false;
}
