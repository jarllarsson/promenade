// =======================================================================================
//                                      FreeImageException
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # FreeImageException
/// Detailed description.....
/// Created on: 5-12-2012 
///---------------------------------------------------------------------------------------
#pragma once
#include <exception>
#include <string>
#include "ToString.h"

using namespace std;

class FreeImageException : public exception
{
public:
	///-----------------------------------------------------------------------------------
	/// Constructor that receives a unique error message
	/// \param p_msg
	/// \param p_file
	/// \param p_func
	/// \param p_line
	/// \return 
	///-----------------------------------------------------------------------------------
	FreeImageException(const string &p_msg,const string &p_file,const string &p_func,
		int p_line){
			m_msg = "\nFreeImage Exception: ";
			m_msg.append(p_msg);
			m_msg.append(" > ");
			m_msg.append(p_file);
			m_msg.append(" : ");
			m_msg.append(p_func);
			m_msg.append(" line ");
			m_msg.append(toString(p_line));
			m_msg.append("\n\n");
	}
	virtual const char* what() const throw()
	{
		return m_msg.c_str();
	}
private:
	string m_msg;
};