#pragma once

#include <BaseException.h>
#include <wtypesbase.h>
#include <string.h>
#include <d3dcommon.h>
using namespace std;

// =======================================================================================
//                                      GraphicsException
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Exception to be thrown by the graphics handler
///        
/// # GraphicsException
/// 
/// 18-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class GraphicsException : public BaseException
{
public:
	GraphicsException(const string& p_msg,
		const string &p_file,const string &p_func,int p_line) 
		: BaseException(Stringify(GraphicsException),p_msg,p_file,p_func,p_line){};

	GraphicsException(HRESULT p_hresult,
		const string &p_file,const string &p_func,int p_line) 
		: BaseException(Stringify(GraphicsException),handleHRESULT(p_hresult),p_file,p_func,p_line){};

	GraphicsException(ID3DBlob* p_errorBlob,
		const string &p_file,const string &p_func,int p_line) 
		: BaseException(Stringify(GraphicsException),handleErrorBlob(p_errorBlob),p_file,p_func,p_line){};

private:
	string handleHRESULT(HRESULT p_hresult)
	{
		string msg=" ";
		LPSTR errTxt = NULL;
		FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL, p_hresult, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&errTxt, 0, NULL);
		if (errTxt!=NULL)
			msg+="("+toString(errTxt)+")";
		else
			msg+="(Unknown HRESULT error)";
		return msg;
	}

	string handleErrorBlob(ID3DBlob* p_errorBlob)
	{
		string smsg=" ";
		char msg[10000];
		char* bfrPtr = (char*)p_errorBlob->GetBufferPointer();
		strcpy_s(msg, sizeof(msg), bfrPtr);
		smsg+="("+toString(msg)+")";
		return smsg;
	}

};