#pragma once

#include <BaseException.h>

// =======================================================================================
//                                ContextException
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Exception to be thrown by the window handler
///        
/// # ContextException
/// 
/// 17-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class ContextException : public BaseException
{
public:
	ContextException(const string& p_msg,
					 const string &p_file,const string &p_func,int p_line) 
					 : BaseException(Stringify(ContextException),p_msg,p_file,p_func,p_line){};
};