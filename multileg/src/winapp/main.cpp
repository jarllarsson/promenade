#define WIN32_LEAN_AND_MEAN
#ifndef _WINDEF_
struct HINSTANCE__;
typedef HINSTANCE__* HINSTANCE;
#endif

#include <vector>
#include "App.h"
#include <DebugPrint.h>
#include <ToString.h>
#include <vld.h>


using namespace std;


// =======================================================================================
//     
//						Multi-legged procedurally animated creatures
//
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Main entry point
///        
/// # main
/// 
/// 17-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR pCmdLine, int nCmdShow)
{
#if defined(_WIN32) && (defined(DEBUG) || defined(_DEBUG))
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	SetThreadAffinityMask(GetCurrentThread(), 1);

	App myApp(hInstance,1280,800);
	myApp.run();

	return 0;
}
