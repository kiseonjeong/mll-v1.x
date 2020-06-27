#include "stdafx.h"
#include "MllUtility.h"

#define TRIM_SPACE " \t\n\v"

namespace mll
{
	string mllutil::ltrim(string str) const
	{
		// Trim a string from the left side
		return str.erase(0, str.find_first_not_of(TRIM_SPACE));
	}

	string mllutil::rtrim(string str) const
	{
		// Trim a string from the right side
		return str.erase(str.find_last_not_of(TRIM_SPACE) + 1);
	}

	string mllutil::trim(const string str) const
	{
		// Trim a string from the both sides
		return rtrim(ltrim(str));
	}

	vector<string> mllutil::split(const string str, const string separator) const
	{
		// Split a string by the separator
		vector<string> result;
		string::size_type i = 0;
		string::size_type j = str.find(separator);
		while (j != string::npos)
		{
			result.push_back(str.substr(i, j - i));
			i = ++j;
			j = str.find(separator, j);
		}
		if (j == string::npos)
		{
			result.push_back(str.substr(i, str.length()));
		}

		return result;
	}
}