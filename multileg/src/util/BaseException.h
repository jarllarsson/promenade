#pragma once

#include <exception>
#include <string>
#include "ToString.h"
using namespace std;

// =======================================================================================
//                                      BaseException
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Base exception class for this project
///        
/// # BaseException
/// 
/// 17-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class BaseException : public exception
{
public:
	BaseException(const string& p_name,const string& p_msg,
				  const string &p_file,const string &p_func,int p_line)
	{
		m_name=p_name;
		compileMessage(p_msg,p_file,p_func,p_line);
	}
	virtual ~BaseException() {}

	virtual const char* what() const throw()
	{
		return m_msg.c_str();
	}
protected:
	string m_msg;
	string m_name;
private:
	void compileMessage(const string &p_msg,const string &p_file,const string &p_func,int p_line)
	{
		m_name="\n\n"+m_name+": ";
		m_msg = m_name+"\n"+p_msg+" [ "+p_file+" : "+p_func+" ("+toString(p_line)+") ]\n\n";
	}

};