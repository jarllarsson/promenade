#pragma once

#include <Windows.h>
#include "ConsoleContext.h"
// =======================================================================================
//                                      DebugPrint
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # DebugPrint
/// 
/// 17-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

/***************************************************************************/
/* FORCE_DISABLE_OUTPUT removes all debug prints silently discarding them  */
/***************************************************************************/
// #define FORCE_DISABLE_OUTPUT


// Will only print in debug, will replace call in release with "nothing"
// call like this: DEBUGPRINT(("text"));
// #ifdef _DEBUG
static void debugPrint(const char* msg);
#ifndef FORCE_DISABLE_OUTPUT
#define DEBUGPRINT(x) debugPrint x
#else
#define DEBUGPRINT(x)
#endif
void debugPrint(const char* msg)
{
	OutputDebugStringA(msg);
	ConsoleContext::addMsg(string(msg), false);
}


// Warning version
// #ifdef _DEBUG
static void debugWarn(const char* msg);
#ifndef FORCE_DISABLE_OUTPUT
#define DEBUGWARNING(x) debugWarn x
#else
#define DEBUGWARNING(x)
#endif
void debugWarn(const char* msg)
{
	OutputDebugStringA((msg));
	MessageBoxA(NULL, (msg), "Warning!", MB_ICONWARNING);
}