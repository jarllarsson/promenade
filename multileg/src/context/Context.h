#pragma once

#include <windows.h>
#include <string>
#include <utility>

using namespace std;

static LRESULT CALLBACK WndProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam );


// =======================================================================================
//                                      Context
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Window context for DirectX
///        
/// # Context
/// 
/// 17-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class Context
{
public:
	Context(HINSTANCE p_hInstance, const string& p_title,
		int p_width, int p_height);
	virtual ~Context();
	HWND getWindowHandle();
	static Context* getInstance();

	void close();

	///-----------------------------------------------------------------------------------
	/// Resize the window
	/// \param p_w
	/// \param p_h
	/// \param p_update set to true to force an update, 
	///					if an update has already been done by windows, set to false.
	/// \return void
	///-----------------------------------------------------------------------------------
	void resize(int p_w, int p_h, bool p_update);

	///-----------------------------------------------------------------------------------
	/// Change the window title string
	/// \param p_title
	/// \return void
	///-----------------------------------------------------------------------------------
	void setTitle(const string& p_title);

	///-----------------------------------------------------------------------------------
	/// Update the window with the store title
	/// \param p_appendMsg Optional message string to append to title
	/// \return void
	///-----------------------------------------------------------------------------------
	void updateTitle(const string& p_appendMsg="");

	///-----------------------------------------------------------------------------------
	/// Whether a closedown was requested
	/// \return bool
	///-----------------------------------------------------------------------------------
	bool closeRequested() const;

	///-----------------------------------------------------------------------------------
	/// Returns true if window has been resized. Dirty bit is reset upon call if true.
	/// \return bool
	///-----------------------------------------------------------------------------------
	bool isSizeDirty();

	pair<int,int> getSize();
protected:
private:
	bool m_closeFlag;
	bool m_sizeDirty;
	string m_title;
	int m_width;
	int m_height;
	HINSTANCE	m_hInstance;
	HWND		m_hWnd;
	static Context* m_instance;
};