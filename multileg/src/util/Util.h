#pragma once

#define SAFE_DELETE(x) if( x ) { delete (x); (x) = NULL; }