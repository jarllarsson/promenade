// =======================================================================================
//                                      UniqueIndexList
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	A list that reuses its indices
///        
/// # UniqueIndexList
/// Detailed description.....
/// Created on: 5-12-2012 (From Amalgamation)
///---------------------------------------------------------------------------------------
#pragma once
#include <stack>
#include <vector>

using namespace std;

template<class T>
class UniqueIndexList
{
public:
	unsigned int add(T p_valueRef);

	bool		 freeIndexAt(unsigned int p_index);

	T			 at(unsigned int p_index);
	T			 operator[](unsigned int p_index);

	bool		 hasValue(unsigned int p_index);

	unsigned int getSize();

	void		 clear();
protected:
private:
	vector<T> m_list;
	stack<unsigned int> m_freeIndices;
};

template<class T>
unsigned int UniqueIndexList<T>::add(T p_valueRef)
{
	unsigned int index = -1;
	if (m_freeIndices.size()>0)
	{
		index = m_freeIndices.top();
		m_list[index] = p_valueRef;
		m_freeIndices.pop();
	}
	else
	{
		m_list.push_back(p_valueRef);
		index = m_list.size()-1;
	}
	return index;
}

template<class T>
bool UniqueIndexList<T>::freeIndexAt(unsigned int p_index)
{
	if (p_index<m_list.size())
	{
		m_list[p_index] = NULL;
		m_freeIndices.push(p_index);
		return true;
	}
	return false;
}

template<class T>
T UniqueIndexList<T>::at(unsigned int p_index)
{
	return m_list[p_index];
}

template<class T>
T UniqueIndexList<T>::operator[](unsigned int p_index)
{
	return m_list[p_index];
}

template<class T>
bool UniqueIndexList<T>::hasValue( unsigned int p_index )
{
	if (p_index < m_list.size())
		return m_list[p_index] != NULL;
	else
		return false;
}

template<class T>
unsigned int UniqueIndexList<T>::getSize()
{
	return m_list.size();
}


template<class T>
void UniqueIndexList<T>::clear()
{
	m_list.clear();
	while(!m_freeIndices.empty())
		m_freeIndices.pop();
}