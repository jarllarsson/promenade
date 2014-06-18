#pragma once
#include <windows.h>
// =======================================================================================
//                                      IContextProcessable
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	interface for object that can process windows events
///        
/// # ContextProcessable
/// 
/// 18-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class IContextProcessable
{
public:
	IContextProcessable() {}
	virtual ~IContextProcessable() {}
	virtual bool processEvent(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) = 0;
protected:
private:
	IContextProcessable(IContextProcessable& p_copy) {}
};