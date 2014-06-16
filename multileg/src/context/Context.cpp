#include "Context.h"
#include "ContextException.h"
#include <algorithm> 

Context* Context::m_instance=NULL;

Context::Context( HINSTANCE p_hInstance, const string& p_title, 
				 int p_width, int p_height )
{
	m_closeFlag=false;
	m_sizeDirty=false;
	m_hInstance = p_hInstance; 
	m_title = p_title;

	// Register class
	WNDCLASSEX wcex;
	wcex.cbSize = sizeof(WNDCLASSEX); 
	wcex.style          = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc    = WndProc; // Callback function
	wcex.cbClsExtra     = 0;
	wcex.cbWndExtra     = 0;
	wcex.hInstance      = m_hInstance;
	wcex.hIcon          = 0;
	wcex.hCursor        = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
	wcex.lpszMenuName   = NULL;
	wcex.lpszClassName  = m_title.c_str();
	wcex.hIconSm        = 0;

	if( !RegisterClassEx(&wcex) )
	{
		throw ContextException("Could not register Context class",
			__FILE__,__FUNCTION__,__LINE__);
	}

	// Create the window
	RECT rc = { 0, 0, p_width, p_height};
	AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );
	if(!(m_hWnd = CreateWindow(
		m_title.c_str(),
		m_title.c_str(),
		WS_OVERLAPPEDWINDOW,
		0,0,
		rc.right - rc.left, rc.bottom - rc.top,
		NULL,NULL,
		m_hInstance,
		NULL)))
	{
		throw ContextException("Could not create window",
			__FILE__,__FUNCTION__,__LINE__);
	}

	ShowWindow( m_hWnd, true );
	ShowCursor(true);
	m_instance=this;
}

Context::~Context()
{
	DestroyWindow(m_hWnd);
}

void Context::setTitle( const string& p_title )
{
	m_title=p_title;
}

void Context::updateTitle( const string& p_appendMsg )
{
	SetWindowText(m_hWnd, (m_title+p_appendMsg).c_str());
}


HWND Context::getWindowHandle()
{
	return m_hWnd;
}

Context* Context::getInstance()
{
	return m_instance;
}

void Context::resize( int p_w, int p_h, bool p_update)
{
	m_width = max(1,p_w);
	m_height = max(1,p_h);
	if (p_update) SetWindowPos( m_hWnd, HWND_TOP, 0, 0, m_width, m_height, SWP_NOMOVE );
	m_sizeDirty=true;
}

bool Context::closeRequested() const
{
	return m_closeFlag;
}

void Context::close()
{
	m_closeFlag=true;
}

pair<int,int> Context::getSize()
{
	return pair<int,int>(m_width,m_height);
}

bool Context::isSizeDirty()
{
	bool isDirty = m_sizeDirty;
	m_sizeDirty=false;
	return isDirty;
}


LRESULT CALLBACK WndProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam )
{
	PAINTSTRUCT ps;
	HDC hdc;

	switch (message) 
	{
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		EndPaint(hWnd, &ps);
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		{
			Context* context = Context::getInstance();
			if (context)
				context->close();
		}
		break;

	case WM_SIZE:
		{
			Context* context = Context::getInstance();
			if (context)
				context->resize(LOWORD(lParam),HIWORD(lParam),false);
		}
		break;

	case WM_KEYDOWN:
		switch(wParam)
		{
		case VK_ESCAPE:
			PostQuitMessage(0);
			Context::getInstance()->close();
			break;
		}
		break;

	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}

	return 0;
}
