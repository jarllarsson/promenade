// =======================================================================================
//                                      RunLengthList
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	A list that defines the number of copies that exist of each entry
///        
/// # RunLengthList
/// Detailed description.....
/// Created on: 28-11-2014
///---------------------------------------------------------------------------------------
#pragma once
#include <vector>

using namespace std;

template<class T>
class RunLengthList
{
public:
	unsigned int add(T p_valueRef, unsigned int p_copies);

	bool		 removeAt(unsigned int p_index);

	T			 at(unsigned int p_index);
	T			 operator[](unsigned int p_index);
	unsigned int runLengthAt(unsigned int p_index);

	bool		 hasValue(unsigned int p_index);

	unsigned int getSize();

	void		 clear();
protected:
private:
	vector<T> m_list;
	vector<unsigned int> m_rllist;
};

template<class T>
unsigned int RunLengthList<T>::add(T p_valueRef, unsigned int p_copies)
{
	unsigned int index = 0;
	m_list.push_back(p_valueRef);
	m_rllist.push_back(p_copies);
	index = (unsigned int)m_list.size() - 1;
	return index;
}

template<class T>
bool RunLengthList<T>::removeAt(unsigned int p_index)
{
	if (p_index < m_list.size())
	{
		m_list.erase(p_index);
		m_rllist.erase(p_index);
		return true;
	}
	return false;
}

template<class T>
T RunLengthList<T>::at(unsigned int p_index)
{
	return m_list[p_index];
}

template<class T>
T RunLengthList<T>::operator[](unsigned int p_index)
{
	return m_list[p_index];
}


template<class T>
unsigned int RunLengthList<T>::runLengthAt(unsigned int p_index)
{
	return m_rllist[p_index];
}


template<class T>
bool RunLengthList<T>::hasValue(unsigned int p_index)
{
	if (p_index < m_list.size())
		return m_list[p_index] != NULL;
	else
		return false;
}

template<class T>
unsigned int RunLengthList<T>::getSize()
{
	return m_list.size();
}


template<class T>
void RunLengthList<T>::clear()
{
	m_list.clear();
	m_rllist.clear();
}