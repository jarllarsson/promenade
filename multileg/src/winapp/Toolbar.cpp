#include "Toolbar.h"
#include <AntTweakBar.h>
#include <ToString.h>


Toolbar::Toolbar(void* p_device) : IContextProcessable()
{
	TwInit(TW_DIRECT3D11, p_device);

	m_bars.push_back(Bar("PLAYER"));
	m_bars.push_back(Bar("PERFORMANCE"));
	m_bars.push_back(Bar("CHARACTER"));

	//TwDefine(" GLOBAL fontscaling=0.5 "); // font size must be set before init (exception)
	init();
}

Toolbar::~Toolbar()
{
	TwWindowSize(0, 0);
	TwTerminate();
}

void Toolbar::init()
{
	TwDefine(" GLOBAL contained=true ");
	TwDefine(" GLOBAL fontsize=1 ");
	TwDefine(" GLOBAL fontresizable=false ");
	defineBarParams(PLAYER, dawnBringerPalRGB[COL_DARKBLUE], " position= '0 0' size='150 150' refresh=0.05");
	defineBarParams(PERFORMANCE, dawnBringerPalRGB[COL_DARKBROWN], " position= '0 150' size='150 150' refresh=0.05");
	defineBarParams(CHARACTER, dawnBringerPalRGB[COL_DARKPURPLE], " position= '0 300' size='150 400' refresh=0.05");
}

bool Toolbar::processEvent(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	if (TwEventWin(hWnd, message, wParam, lParam)) // send event message to AntTweakBar
		return true; // event has been handled by AntTweakBar
	return false;
}


void Toolbar::setWindowSize(int p_width, int p_height)
{
	TwWindowSize(p_width, p_height);
}

void Toolbar::draw()
{
	TwDraw();
}

void Toolbar::defineBarParams(BarType p_type, const char* p_params)
{
	std::string tmp = " ";
	tmp += m_bars[(int)p_type].m_name;
	tmp += p_params;
	TwDefine(tmp.c_str());
}


void Toolbar::defineBarParams(BarType p_type, const Color3f& p_color, const char* p_params)
{
	std::string colStr = " color='";
	colStr += ToString((int)(256.0f*p_color.r)) + " ";
	colStr += ToString((int)(256.0f*p_color.g)) + " ";
	colStr += ToString((int)(256.0f*p_color.b));
	colStr += "' alpha=128 ";
	defineBarParams(p_type, (colStr + std::string(p_params)).c_str());
}
void Toolbar::defineBarParams(BarType p_type, float p_min, float p_max, float p_stepSz, const char* p_params)
{
	std::string paramStr = " min='";
	paramStr += ToString(p_min) + " ";
	paramStr += "' max=";
	paramStr += ToString(p_max) + " ";
	paramStr += "' step=";
	paramStr += ToString(p_stepSz);
	paramStr += "' ";
	defineBarParams(p_type, (paramStr + std::string(p_params)).c_str());
}

void Toolbar::defineBarParams(BarType p_type, int p_min, int p_max, const char* p_params)
{
	std::string paramStr = " min='";
	paramStr += ToString(p_min) + " ";
	paramStr += "' max=";
	paramStr += ToString(p_max);
	paramStr += "' ";
	defineBarParams(p_type, (paramStr + std::string(p_params)).c_str());
}

void Toolbar::addReadOnlyVariable(BarType p_barType, const char* p_name, VarType p_type, const void *p_var, const char* p_misc/*=""*/)
{
	TwAddVarRO(getBar(p_barType), p_name, (TwType)p_type, p_var, p_misc);
}

void Toolbar::addReadWriteVariable(BarType p_barType, const char* p_name, VarType p_type, void *p_var, const char* p_misc/*=""*/)
{
	TwAddVarRW(getBar(p_barType), p_name, (TwType)p_type, p_var, p_misc);
}

void Toolbar::addSeparator(BarType p_barType, const char* p_name, const char* p_misc/*=""*/)
{
	TwAddSeparator(getBar(p_barType), p_name, p_misc);
}

void Toolbar::addButton(BarType p_barType, const char* p_name, TwButtonCallback p_callback, void *p_inputData, const char* p_misc/*=""*/)
{
	TwAddButton(getBar(p_barType), p_name, p_callback, p_inputData, p_misc);
}

void Toolbar::addLabel(BarType p_barType, const char* p_name, const char* p_misc/*=""*/)
{
	TwAddButton(getBar(p_barType), p_name, NULL, NULL, p_misc);
		//(std::string(" label='")+std::string(p_label)+"'").c_str() );
}

TwBar* Toolbar::getBar(BarType p_type)
{
	return m_bars[(int)p_type].m_bar;
}

void Toolbar::clearBar(BarType p_barType)
{
	TwRemoveAllVars(getBar(p_barType));
}

void TW_CALL boolButton(void* p_bool)
{
	*(bool*)p_bool = !*(bool*)p_bool;
}
