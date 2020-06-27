#include "stdafx.h"
#include "MllDataset.h"

namespace mll
{
	mlldata::mlldata() : X(_X), T(_T), dimension(_dimension), nsample(_nsample), C(_C), nclass(_nclass), slope(_slope), intercept(_intercept)
	{
		// Set an object
		setObject();
	}

	mlldata::mlldata(const string path, const string separator, const labelpos mode) : X(_X), T(_T), dimension(_dimension), nsample(_nsample), C(_C), nclass(_nclass), slope(_slope), intercept(_intercept)
	{
		// Set an object
		setObject();

		// Open a dataset
		open(path, separator, mode);
	}

	mlldata::mlldata(const numat& X, const numat& T) : X(_X), T(_T), dimension(_dimension), nsample(_nsample), C(_C), nclass(_nclass), slope(_slope), intercept(_intercept)
	{
		// Set an object
		setObject();

		// Set the dataset
		set(X, T);
	}

	mlldata::mlldata(const numat& X) : X(_X), T(_T), dimension(_dimension), nsample(_nsample), C(_C), nclass(_nclass), slope(_slope), intercept(_intercept)
	{
		// Set an object
		setObject();

		// Set the dataset
		set(X);
	}

	mlldata::mlldata(const mlldata& obj) : X(_X), T(_T), dimension(_dimension), nsample(_nsample), C(_C), nclass(_nclass), slope(_slope), intercept(_intercept)
	{
		// Clone the object
		*this = obj;
	}

	mlldata::~mlldata()
	{
		// Clear the object
		clearObject();
	}

	mlldata& mlldata::operator=(const mlldata& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	const numat mlldata::operator[](const int idx) const
	{
		// Check the index
		assert(idx >= 0 && idx < 3);

		// Get the vector matrix
		switch (idx)
		{
		case 0: return _X;			// Feature vector
		case 1: return _T;			// Target vector
		case 2: return _C;			// Class vector
		default: return numat();
		}
	}

	void mlldata::setObject()
	{
		// Set the parameters
		setType(*this);
		_dimension = -1;
		_nsample = -1;
		_nclass = -1;

		// Clear the memories
		_X.release();
		_T.release();
		_C.release();
	}

	void mlldata::copyObject(const object& obj)
	{
		// Do down casting
		mlldata* _obj = (mlldata*)&obj;

		// Copy the parameters
		_dimension = _obj->_dimension;
		_nsample = _obj->_nsample;
		_nclass = _obj->_nclass;

		// Copy the memories
		_X = _obj->_X;
		_T = _obj->_T;
		_C = _obj->_C;
	}

	void mlldata::clearObject()
	{
		// Clear the memories
		_X.release();
		_T.release();
		_C.release();
	}

	const int mlldata::open(const string path, const string separator, const labelpos mode)
	{
		// Create a dataset reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Read the dataset
		vector<numat> datasetX;
		vector<numat> datasetT;
		string lineStr;
		string trimStr;
		vector<string> splitStr;
		bool firstData = true;
		while (!reader.eof())
		{
			// Read and check a string value
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			splitStr = split(trimStr, separator);
			if (trimStr == "" || splitStr.size() == 1)
			{
				continue;
			}

			// Check the label position
			if (mode == LABEL_REAR || mode == LABEL_FRONT)
			{
				// Check a first string value
				if (firstData == true)
				{
					// Initialize a feature dimension
					_dimension = (int)splitStr.size() - 1;
					firstData = false;
				}
				else
				{
					// Check a feature dimension
					assert(splitStr.size() - 1 == _dimension);
					if (splitStr.size() - 1 != _dimension)
					{
						return -1;
					}
				}

				// Check the label position
				numat sampleX(msize(1, _dimension));
				numat sampleT(msize(1, 1));
				if (mode == LABEL_REAR)
				{
					for (int i = 0; i < _dimension; i++)
					{
						sampleX(i) = atof(splitStr[i].c_str());
					}
					sampleT(0) = atof(splitStr[_dimension].c_str());
				}
				else
				{
					for (int i = 1; i < 1 + _dimension; i++)
					{
						sampleX(i - 1) = atof(splitStr[i].c_str());
					}
					sampleT(0) = atof(splitStr[0].c_str());
				}

				// Save the dataset
				datasetX.push_back(sampleX);
				datasetT.push_back(sampleT);
			}
			else
			{
				// Check a first string value
				if (firstData == true)
				{
					// Initialize a feature dimension
					_dimension = (int)splitStr.size();
					firstData = false;
				}
				else
				{
					// Check a feature dimension
					assert(splitStr.size() == _dimension);
					if (splitStr.size() != _dimension)
					{
						return -1;
					}
				}

				// Check the label position
				numat sampleX(msize(1, _dimension));
				for (int i = 0; i < _dimension; i++)
				{
					sampleX(i) = atof(splitStr[i].c_str());
				}

				// Save the dataset
				datasetX.push_back(sampleX);
				datasetT.push_back(sampleX);
			}
		}
		reader.close();

		// Set the dataset
		_X = numat::vappend(datasetX);
		_T = numat::vappend(datasetT);

		// Check the label position
		if (mode == LABEL_REAR || mode == LABEL_FRONT)
		{
			// Set information
			_dimension = _X.cols;
			_nsample = _X.rows;
			_C = _T.uniq();
			_nclass = _C.length();
		}
		else
		{
			// Set information
			_dimension = _X.cols;
			_nsample = _X.rows;
			_C = numat::zeros(msize(1));
			_nclass = _C.length();
		}

		return 0;
	}

	void mlldata::set(const numat& X, const numat& T)
	{
		// Check the vector length
		assert(X.rows == T.rows);

		// Set the dataset
		_X = X;
		_T = T;

		// Set information
		_dimension = _X.cols;
		_nsample = _X.rows;
		_C = _T.uniq();
		_nclass = _C.length();
	}

	void mlldata::set(const numat& X)
	{
		// Set the dataset
		_X = X;
		_T = X;

		// Set information
		_dimension = _X.cols;
		_nsample = _X.rows;
		_C = numat::zeros(msize(1));
		_nclass = _C.length();
	}

	const bool mlldata::empty() const
	{
		// Check the dataset is empty or not
		if (_X.empty() == true || _T.empty() == true)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	void mlldata::shuffle(const int iter)
	{
		// Check a sample size
		if (_X.rows > 1)
		{
			// Set a iteration for shuffling
			int maxIter = iter;
			if (iter < 0)
			{
				maxIter = _X.rows;
			}

			// Shuffle the dataset
			int count = 0;
			while (count < maxIter)
			{
				// Randomly select an index
				random_device rd;
				mt19937 gen(rd());
				uniform_int_distribution<int> dist(0, _X.rows - 1);
				int i = dist(gen);
				int j = i;
				while (j == i)
				{
					j = dist(gen);
				}

				// Swap the sample data
				_X.swapr(i, j);
				_T.swapr(i, j);
				count++;
			}
		}
	}

	void mlldata::scale()
	{
		// Create the vector memories for the feature scaling
		_slope = numat::zeros(msize(1, _X.cols));
		_intercept = numat::zeros(msize(1, _X.cols));

		// Normalize the dataset using the feature scaling
		const double epsilon = 1e-5;
		for (int i = 0; i < _X.cols; i++)
		{
			// Find the min, max values
			double minVal = _X(0, i);
			double maxVal = _X(0, i);
			for (int j = 0; j < _X.rows; j++)
			{
				if (_X(j, i) < minVal)
				{
					minVal = _X(j, i);
				}
				if (_X(j, i) > maxVal)
				{
					maxVal = _X(j, i);
				}
			}

			// Normalize the feature value
			const double denom = max(maxVal - minVal, epsilon);
			for (int j = 0; j < _X.rows; j++)
			{
				_X(j, i) = (_X(j, i) - minVal) / denom;
			}

			// Save the scaling value
			_slope(i) = denom;
			_intercept(i) = minVal;
		}
	}

	void mlldata::normalize()
	{
		// Create the vector memories for the standard score
		_slope = numat::zeros(msize(1, _X.cols));
		_intercept = numat::zeros(msize(1, _X.cols));

		// Normalize the dataset using the standard score
		const double epsilon = 1e-5;
		for (int i = 0; i < _X.cols; i++)
		{
			// Calculate a mean value
			double mean = 0.0;
			for (int j = 0; j < _X.rows; j++)
			{
				mean += _X(j, i);
			}
			mean /= _X.rows;

			// Calculate a variance value
			double var = 0.0;
			for (int j = 0; j < _X.rows; j++)
			{
				var += (_X(j, i) - mean) * (_X(j, i) - mean);
			}
			var /= _X.rows;

			// Normalize the feature value
			const double denom = max(sqrt(var), epsilon);
			for (int j = 0; j < _X.rows; j++)
			{
				_X(j, i) = (_X(j, i) - mean) / denom;
			}

			// Save the scaling value
			_slope(i) = denom;
			_intercept(i) = mean;
		}
	}

	const vector<mlldata> mlldata::subdata(const int subsize, const bool shuffling) const
	{
		// Check the dataset is empty or not
		assert(empty() == false);

		// Check a size of the dataset
		assert(subsize > 0);

		// Initialize the parameters
		int nsub = (int)ceil((double)_nsample / subsize);

		// Set sub-dataset length information
		numem<int> length(msize(nsub, 1), 0);
		for (int i = 0; i < nsub; i++)
		{
			if (_nsample / (subsize * (i + 1)) > 0)
			{
				length(i) = subsize;
			}
			else
			{
				length(i) = _nsample % subsize;
			
			}
		}

		// Check the shuffling flag
		numem<int> index(msize(_nsample, 1), -1);
		if (shuffling == true && _nclass > 1)
		{
			// Set sub-dataset index information
			numem<bool> flag(msize(_nsample, 1), false);
			int count = 0;
			while (count < _nsample)
			{
				int target = count;
				while (true)
				{
					bool selection = false;
					for (int i = 0; i < _nsample; i++)
					{
						if (flag(i) == false && _T(i) == _C(target % _nclass))
						{
							flag(i) = true;
							index(count) = i;
							selection = true;
							count++;
							break;
						}
					}
					if (selection == false)
					{
						target++;
					}
					else
					{
						break;
					}
				}
			}
		}
		else
		{
			// Set sub-dataset index information
			for (int i = 0; i < _nsample; i++)
			{
				index(i) = i;
			}
		}
		
		// Generate the sub-dataset
		vector<mlldata> subset;
		if (_nclass > 1)
		{
			for (int i = 0; i < nsub; i++)
			{
				numat subX(msize(length(i), _dimension));
				for (int j = 0, l = i * subsize; j < subX.rows; j++, l++)
				{
					for (int k = 0; k < subX.cols; k++)
					{
						subX(j, k) = _X(index(l), k);
					}
				}
				numat subT(msize(length(i), 1));
				for (int j = 0, l = i * subsize; j < subT.rows; j++, l++)
				{
					for (int k = 0; k < subT.cols; k++)
					{
						subT(j, k) = _T(index(l), k);
					}
				}
				subset.push_back(mlldata(subX, subT));
			}
		}
		else
		{
			for (int i = 0; i < nsub; i++)
			{
				numat subX(msize(length(i), _dimension));
				for (int j = 0, l = i * subsize; j < subX.rows; j++, l++)
				{
					for (int k = 0; k < subX.cols; k++)
					{
						subX(j, k) = _X(index(l), k);
					}
				}
				subset.push_back(mlldata(subX));
			}
		}

		// Check the shuffling flag
		if (shuffling == true)
		{
			for (int i = 0; i < nsub; i++)
			{
				subset[i].shuffle();
			}
		}

		return subset;
	}
}